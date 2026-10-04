#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Fetch an arm64 Linux kernel and make a busybox initramfs, for ruvm's Linux boot test.

The kernel is the netboot one of Debian 12 (an uncompressed arm64 Image with the PL011,
GICv3, PSCI and generic timer drivers built in). The initramfs holds the static busybox of
Debian 12 and an /init that mounts /proc, /sys and /dev and starts a shell on the console.
The cpio archive is written here, so making it needs no root.

Usage:
    scripts/arm64-linux-test-image.py OUTDIR

and then

    RUVM_TEST_ARM64_KERNEL=OUTDIR/linux RUVM_TEST_ARM64_INITRD=OUTDIR/initramfs.cpio.gz \\
        cargo test -p ruvm-cli --release --test arm_virt -- --ignored linux_boots_to_a_shell

or by hand:

    qemu-system-aarch64 -M virt -cpu max -smp 2 -m 512 -nographic -kernel OUTDIR/linux \\
        -initrd OUTDIR/initramfs.cpio.gz -append console=ttyAMA0

The images stay out of the tree.
"""

import gzip
import io
import lzma
import os
import sys
import tarfile
import urllib.request

DEBIAN = "http://deb.debian.org/debian"
KERNEL = DEBIAN + (
    "/dists/bookworm/main/installer-arm64/current/images/netboot/debian-installer/arm64/linux"
)
BUSYBOX = DEBIAN + "/pool/main/b/busybox/busybox-static_1.35.0-4+deb12u1+b1_arm64.deb"

INIT = b"""#!/bin/busybox sh
/bin/busybox --install -s /bin
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev 2>/dev/null
echo "ruvm initramfs: $(uname -sr) on $(grep -c ^processor /proc/cpuinfo) cpus"
exec setsid cttyhack sh
"""


def fetch(url):
    print(f"fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as r:
        return r.read()


def deb_member(deb, prefix):
    """The contents of the ar member of a .deb whose name starts with prefix."""
    assert deb[:8] == b"!<arch>\n", "not an ar archive"
    off = 8
    while off < len(deb):
        name = deb[off:off + 16].decode().strip()
        size = int(deb[off + 48:off + 58].decode().strip())
        data = deb[off + 60:off + 60 + size]
        if name.startswith(prefix):
            return name, data
        off += 60 + size + (size & 1)
    raise SystemExit(f"no {prefix} in the package")


def busybox_binary(deb):
    name, data = deb_member(deb, "data.tar")
    if name.endswith(".xz"):
        data = lzma.decompress(data)
    elif name.endswith(".gz"):
        data = gzip.decompress(data)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        return t.extractfile("./bin/busybox").read()


def cpio_newc(entries):
    """A newc cpio archive of (name, mode, rdev major, rdev minor, data) entries."""
    out = bytearray()
    ino = 1

    def add(name, mode, major, minor, data):
        nonlocal ino
        name_b = name.encode() + b"\0"
        fields = [ino, mode, 0, 0, 1, 0, len(data), 0, 0, major, minor, len(name_b), 0]
        out.extend(b"070701" + b"".join(b"%08X" % f for f in fields))
        out.extend(name_b)
        out.extend(b"\0" * (-len(out) % 4))
        out.extend(data)
        out.extend(b"\0" * (-len(out) % 4))
        ino += 1

    for e in entries:
        add(*e)
    add("TRAILER!!!", 0, 0, 0, b"")
    return bytes(out)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    kernel = os.path.join(outdir, "linux")
    if not os.path.exists(kernel):
        data = fetch(KERNEL)
        with open(kernel, "wb") as f:
            f.write(data)
    busybox = busybox_binary(fetch(BUSYBOX))
    d, f, c = 0o040755, 0o100755, 0o020600
    entries = [
        ("bin", d, 0, 0, b""),
        ("dev", d, 0, 0, b""),
        ("etc", d, 0, 0, b""),
        ("proc", d, 0, 0, b""),
        ("sys", d, 0, 0, b""),
        ("bin/busybox", f, 0, 0, busybox),
        ("init", f, 0, 0, INIT),
        ("dev/console", c, 5, 1, b""),
        ("dev/null", c | 0o066, 1, 3, b""),
    ]
    initrd = os.path.join(outdir, "initramfs.cpio.gz")
    with open(initrd, "wb") as out:
        out.write(gzip.compress(cpio_newc(entries), 9))
    print(f"RUVM_TEST_ARM64_KERNEL={kernel}")
    print(f"RUVM_TEST_ARM64_INITRD={initrd}")


if __name__ == "__main__":
    main()
