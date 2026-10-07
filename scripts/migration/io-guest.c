// SPDX-License-Identifier: GPL-2.0-or-later
//
// A Linux /init that keeps virtio disks, NICs and the entropy device busy, for live migration
// tests. Build it with "gcc -static -O2" and boot it as the only file of an initramfs, see
// build-io-guest.sh.
//
// Every step, about a second apart, it
//
// - writes 64 blocks of 4 KiB with O_DIRECT to /dev/vda (virtio-blk) and /dev/sda (a disk on
//   virtio-scsi) where present, at a slot that cycles through the first 32 MiB, and reads back
//   and checks the slots written 1, 37 and 100 steps before, so data written before a migration
//   is checked after it;
// - sends frames of ethertype 0x88b5 out of eth0 and receives them on eth1, the two NICs being
//   joined by a hub, and checks the payload of each frame received;
// - reads 16 bytes from /dev/hwrng where present, first loading /virtio-rng.ko.zst or
//   /virtio-rng.ko when the initramfs has it, for kernels with the driver as a module;
//
// and then prints "IO STEP BLOCKS TX RX BAD OK" on /dev/console, all in hex: STEP goes up by
// one, BLOCKS counts the blocks checked, TX and RX the frames sent and received, BAD the
// failures. A failure also prints a line ending in "BAD REASON". The first line is
// "IO START" with the devices found. It never exits.

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <poll.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mount.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define BLOCK 4096
#define RUN 64                     // blocks written per step
#define SLOTS (32 * 256 / RUN)     // 32 MiB of runs
#define ETHERTYPE 0x88b5
#define PAYLOAD 256

static int con = 2;
static uint32_t step, blocks, tx, rx, bad;

static void say(const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > (int)sizeof(buf) - 1) {
        n = sizeof(buf) - 1;
    }
    // One write per line, so kernel messages do not land in the middle of it.
    if (write(con, buf, n) < 0) {
        // Nowhere to report it.
    }
}

static void fail(const char *fmt, ...)
{
    char why[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(why, sizeof(why), fmt, ap);
    va_end(ap);
    bad++;
    say("IO %08x %08x %08x %08x %08x BAD %s\n", step, blocks, tx, rx, bad, why);
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static uint32_t mix(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

// Fills buf with the run of disk dev written at step s. Each block gets its own seed from
// (dev, s, block) and its words a cheap sequence from that seed, since TCG makes a full hash
// per word cost most of a step.
static void fill(uint32_t *buf, uint32_t dev, uint32_t s)
{
    uint32_t first = (s % SLOTS) * RUN;
    for (uint32_t b = 0; b < RUN; b++) {
        uint32_t seed = mix(dev * 0x9e3779b1 ^ mix(s * 0x27d4eb2f ^ mix(first + b)));
        for (uint32_t i = 0; i < BLOCK / 4; i++) {
            buf[b * (BLOCK / 4) + i] = seed ^ (i * 0x165667b1);
        }
    }
}

struct disk {
    const char *path;
    int fd;
};

static struct disk disks[] = { { "/dev/vda", -1 }, { "/dev/sda", -1 } };
#define NDISKS (sizeof(disks) / sizeof(disks[0]))

static void disk_step(uint32_t *buf, uint32_t *want)
{
    for (uint32_t d = 0; d < NDISKS; d++) {
        int fd = disks[d].fd;
        if (fd < 0) {
            continue;
        }
        fill(buf, d, step);
        off_t off = (off_t)(step % SLOTS) * RUN * BLOCK;
        ssize_t n = pwrite(fd, buf, RUN * BLOCK, off);
        if (n != RUN * BLOCK) {
            fail("%s write at %llx: %zd %s", disks[d].path, (long long)off, n, strerror(errno));
            continue;
        }
        // A flush costs a host fdatasync, so only every 8th step.
        if (step % 8 == 7 && fdatasync(fd) < 0) {
            fail("%s flush: %s", disks[d].path, strerror(errno));
        }
        static const uint32_t back[] = { 1, 37, 100 };
        for (uint32_t k = 0; k < sizeof(back) / sizeof(back[0]); k++) {
            if (step < back[k]) {
                continue;
            }
            uint32_t s = step - back[k];
            off = (off_t)(s % SLOTS) * RUN * BLOCK;
            n = pread(fd, buf, RUN * BLOCK, off);
            if (n != RUN * BLOCK) {
                fail("%s read at %llx: %zd %s", disks[d].path, (long long)off, n, strerror(errno));
                continue;
            }
            fill(want, d, s);
            for (uint32_t b = 0; b < RUN; b++) {
                if (memcmp(buf + b * (BLOCK / 4), want + b * (BLOCK / 4), BLOCK)) {
                    fail("%s block %x of step %x", disks[d].path, (s % SLOTS) * RUN + b, s);
                } else {
                    blocks++;
                }
            }
        }
    }
}

static int txs = -1, rxs = -1;
static unsigned char txmac[6];
static int txif;

static int open_nic(const char *name, unsigned char *mac, int *ifindex)
{
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, name, IFNAMSIZ - 1);
    if (s < 0 || ioctl(s, SIOCGIFFLAGS, &ifr) < 0) {
        if (s >= 0) {
            close(s);
        }
        return -1;
    }
    ifr.ifr_flags |= IFF_UP;
    if (ioctl(s, SIOCSIFFLAGS, &ifr) < 0 || ioctl(s, SIOCGIFHWADDR, &ifr) < 0) {
        close(s);
        return -1;
    }
    if (mac) {
        memcpy(mac, ifr.ifr_hwaddr.sa_data, 6);
    }
    close(s);
    int p = socket(AF_PACKET, SOCK_RAW, htons(ETHERTYPE));
    if (p < 0) {
        return -1;
    }
    struct sockaddr_ll sll;
    memset(&sll, 0, sizeof(sll));
    sll.sll_family = AF_PACKET;
    sll.sll_protocol = htons(ETHERTYPE);
    sll.sll_ifindex = if_nametoindex(name);
    if (bind(p, (struct sockaddr *)&sll, sizeof(sll)) < 0) {
        close(p);
        return -1;
    }
    *ifindex = sll.sll_ifindex;
    fcntl(p, F_SETFL, O_NONBLOCK);
    return p;
}

static void payload(unsigned char *p, uint32_t seq)
{
    for (uint32_t i = 0; i < PAYLOAD; i += 4) {
        uint32_t w = mix(seq * 0x9e3779b1 + i);
        memcpy(p + i, &w, 4);
    }
}

static void send_frame(void)
{
    unsigned char f[14 + 4 + PAYLOAD];
    memset(f, 0xff, 6);
    memcpy(f + 6, txmac, 6);
    f[12] = ETHERTYPE >> 8;
    f[13] = ETHERTYPE & 0xff;
    uint32_t seq = htonl(tx);
    memcpy(f + 14, &seq, 4);
    payload(f + 18, tx);
    struct sockaddr_ll sll;
    memset(&sll, 0, sizeof(sll));
    sll.sll_family = AF_PACKET;
    sll.sll_ifindex = txif;
    sll.sll_halen = 6;
    memset(sll.sll_addr, 0xff, 6);
    if (sendto(txs, f, sizeof(f), 0, (struct sockaddr *)&sll, sizeof(sll)) == sizeof(f)) {
        tx++;
    } else if (errno != ENOBUFS && errno != EAGAIN) {
        fail("send: %s", strerror(errno));
    }
}

static void drain(void)
{
    unsigned char f[2048], want[PAYLOAD];
    for (;;) {
        struct sockaddr_ll from;
        socklen_t len = sizeof(from);
        ssize_t n = recvfrom(rxs, f, sizeof(f), 0, (struct sockaddr *)&from, &len);
        if (n < 0) {
            return;
        }
        if (from.sll_pkttype == PACKET_OUTGOING) {
            continue;
        }
        uint32_t seq;
        if (n < 14 + 4 + PAYLOAD) {
            fail("short frame of %zd bytes", n);
            continue;
        }
        memcpy(&seq, f + 14, 4);
        seq = ntohl(seq);
        payload(want, seq);
        if (memcmp(f + 18, want, PAYLOAD)) {
            fail("frame %x payload", seq);
        } else {
            rx++;
        }
    }
}

// Sends a frame every 10 ms and takes in what arrives, until end.
static void net_until(double end)
{
    double t, send = now();
    while ((t = now()) < end) {
        if (t >= send) {
            if (txs >= 0) {
                send_frame();
            }
            send = t + 0.01;
        }
        int wait = (send - t) * 1000 + 1;
        if (rxs >= 0) {
            struct pollfd p = { rxs, POLLIN, 0 };
            poll(&p, 1, wait);
            drain();
        } else {
            usleep(wait * 1000);
        }
    }
}

// Opens /dev/hwrng if it has a source behind it.
static int open_rng(void)
{
    static const char *const mods[] = { "/virtio-rng.ko.zst", "/virtio-rng.ko" };
    for (int i = 0; i < 2; i++) {
        int fd = open(mods[i], O_RDONLY);
        if (fd >= 0) {
            // MODULE_INIT_COMPRESSED_FILE is 4.
            if (syscall(SYS_finit_module, fd, "", i == 0 ? 4 : 0) < 0 && errno != EEXIST) {
                say("IO START cannot load %s: %s\n", mods[i], strerror(errno));
            }
            close(fd);
            break;
        }
    }
    int fd = open("/dev/hwrng", O_RDONLY);
    unsigned char r[16];
    if (fd >= 0 && read(fd, r, sizeof(r)) <= 0) {
        close(fd);
        fd = -1;
    }
    return fd;
}

int main(void)
{
    mkdir("/dev", 0755);
    mkdir("/proc", 0755);
    mkdir("/sys", 0755);
    mount("devtmpfs", "/dev", "devtmpfs", 0, NULL);
    mount("proc", "/proc", "proc", 0, NULL);
    mount("sysfs", "/sys", "sysfs", 0, NULL);
    int c = open("/dev/console", O_WRONLY | O_NOCTTY);
    if (c >= 0) {
        con = c;
    }

    // Disks behind virtio-scsi show up after an asynchronous scan.
    double give_up = now() + 3;
    for (uint32_t d = 0; d < NDISKS; d++) {
        while ((disks[d].fd = open(disks[d].path, O_RDWR | O_DIRECT)) < 0 && now() < give_up) {
            usleep(100000);
        }
    }
    unsigned char rxmac[6];
    int rxif;
    txs = open_nic("eth0", txmac, &txif);
    rxs = open_nic("eth1", rxmac, &rxif);
    int rng = open_rng();

    char found[128] = "";
    for (uint32_t d = 0; d < NDISKS; d++) {
        if (disks[d].fd >= 0) {
            strcat(found, " ");
            strcat(found, disks[d].path + 5);
        }
    }
    strcat(found, txs >= 0 ? " eth0" : "");
    strcat(found, rxs >= 0 ? " eth1" : "");
    strcat(found, rng >= 0 ? " hwrng" : "");
    say("IO START%s\n", found);

    uint32_t *buf, *want;
    if (posix_memalign((void **)&buf, BLOCK, RUN * BLOCK) ||
        posix_memalign((void **)&want, BLOCK, RUN * BLOCK)) {
        say("IO START BAD no memory\n");
        for (;;) {
            pause();
        }
    }

    // Give the links a moment to come up before the first frames.
    net_until(now() + 0.5);
    tx = rx = 0;
    double next = now();
    for (step = 0;; step++) {
        next += 1;
        disk_step(buf, want);
        if (rng >= 0) {
            unsigned char r[16];
            if (read(rng, r, sizeof(r)) <= 0) {
                fail("hwrng: %s", strerror(errno));
            }
        }
        net_until(next);
        // Catch up without bursts when a step ran long, as after a migration.
        if (now() > next + 1) {
            next = now();
        }
        say("IO %08x %08x %08x %08x %08x OK\n", step, blocks, tx, rx, bad);
    }
}
