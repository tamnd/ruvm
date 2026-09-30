# 15. Networking, character devices, UI and audio

This document covers the host-facing backends that sit behind guest devices: network backends and filters (QEMU's net/), character devices (chardev/), displays and input (ui/), and audio (audio/). The reference is QEMU 11.1.0. The compatibility contract (document 02) is mostly on the command line and QMP side: `-netdev`, `-nic`, `-chardev`, `-display`, `-vnc`, `-spice`, `-audiodev`, the enums and commands in qapi/net.json, char.json, ui.json and audio.json, and the few places where backend state reaches the migration stream (virtio-net's announce state, the qemu-vdagent chardev, audio device state). Everything else is free to change.

## Crates and threads

The canon crates are `ruvm-net` (netdev backends and filters), `ruvm-chardev`, `ruvm-ui` and `ruvm-audio`, all L3 and GPL-2.0-or-later because they port QEMU logic (VNC encoders, the mixing engine, the vt100 emulator, COLO compare). Guest NIC, UART, display and sound models live in `ruvm-hw-net`, `ruvm-hw-char`, `ruvm-hw-display` and `ruvm-hw-audio` and talk to backends only through the canon traits `NetBackend`, `CharBackend`, `DisplayListener` and `AudioBackend`. Frontends never name a backend type.

QEMU runs almost all of this on the main loop under the BQL, with glib `GSource` watches for chardevs, `qemu_set_fd_handler` for tap and sockets, and `GMainContext` integration for GTK. ruvm does not use glib in the core. Each backend is registered on a ruvm-aio reactor: by default the main thread's reactor, or an iothread's reactor when the frontend is bound to one. Toolkits that insist on owning a thread get one: GTK runs on a dedicated UI thread with its own glib main loop, and Cocoa runs on the process main thread (AppKit requires it), which pushes ruvm's own main reactor to a second thread exactly as QEMU's `qemu_main` hook does in ui/cocoa.m. UI threads never touch device state; they receive frames and send input events over bounded channels.

## Networking

### The client model

QEMU's net layer (net/net.c, include/net/net.h) connects `NetClientState` peers. A NIC is a client, a backend is a client, and each has at most one peer. `NetClientInfo` holds the callbacks: `receive`, `receive_iov`, `can_receive`, `link_status_changed`, `query_rx_filter`, `has_vnet_hdr`, `set_offload`, `set_vnet_hdr_len`, `announce`, `set_steering_ebpf`, `check_peer_type`. Packets cross from one peer to the other through a `NetQueue` (net/queue.c) that holds packets when the receiver cannot accept them, up to `nq_maxlen` 10000 packets, after which a sender without a completion callback has its packet dropped, and a sender with a callback (`qemu_send_packet_async`) is told to stop until the callback fires. Filters (below) sit on the path between the two peers.

ruvm keeps the peer graph, because QMP exposes it (`query-rx-filter`, `set_link`, `netdev_del`, the `nic` and `netdev` properties, `info network` output that libvirt tests parse). The trait is batch oriented:

```rust
pub trait NetBackend: Send + Sync {
    fn kind(&self) -> NetClientDriver;
    fn vnet_hdr(&self) -> VnetHdrCaps;                 // has_vnet_hdr, supported lengths
    fn set_vnet_hdr_len(&self, len: usize) -> Result<(), NetError>;
    fn set_offload(&self, ol: &NetOffloads) -> Result<(), NetError>;   // csum, tso4/6, ecn, ufo, uso4/6, tunnel
    fn transmit(&self, pkts: &mut TxBatch<'_>) -> TxStatus;           // frames from the NIC side
    fn attach_rx(&self, sink: RxSink, reactor: ReactorId) -> Result<(), NetError>;
    fn link_changed(&self, up: bool) {}
    fn set_steering_ebpf(&self, prog: Option<BorrowedFd<'_>>) -> Result<(), NetError> { Err(NetError::Unsupported) }
    fn datapath(&self) -> Datapath;                   // InProcess, VhostKernel, VhostUser, VhostVdpa
}
```

`TxBatch` carries up to 64 frames as iovec lists pointing into guest memory (document 13 describes how virtio-net builds them); the backend returns how many it consumed and whether it wants a writability callback, which is the `sent_cb` protocol in batch form. `RxSink` is the NIC side; a backend pushes frames to it and gets back `Accepted(n)` or `Full`. When the NIC reports `Full`, ruvm stops reading the host fd (tap, socket, AF_XDP RX ring) instead of queueing, so the kernel's queue is the buffer and back pressure reaches the host side. QEMU's tap also stops polling when the peer cannot receive; ruvm drops the intermediate `NetQueue` copy. The `NetQueue` itself still exists for the paths where QEMU semantics require it: hubs, filters that hold packets, and backends that cannot be paused (dgram sockets, slirp), where it keeps the same 10000 packet limit so drop behavior under overload matches.

Offloads: the vnet header (`struct virtio_net_hdr_v1_hash` with its 10, 12 or 20 byte variants) passes through from virtio-net to tap and to AF_XDP-less paths unchanged when both peers support it; `set_offload` maps to `TUNSETOFFLOAD` on tap. QEMU 11.1 extends `NetOffloads` with UDP tunnel GSO for virtio-net's `host_tunnel` and `guest_tunnel` features, backed by the Linux `TUN_F_UDP_TUNNEL_GSO` flags; ruvm offers the same set when the tap device reports them and nothing more.

### Hubs and the legacy syntax

The legacy `-net nic -net user` form attaches clients to hub 0 (net/hub.c) through `hubport` clients; a hub floods each frame to every port except the source. `-nic` creates a netdev and NIC pair directly, without a hub. ruvm implements hubs as a small in-process switch with QEMU's port naming (`hub0port0`), because `info network` prints it, and clones a frame only for the second and later ports.

### Backends

| Netdev | QEMU source | Hosts | ruvm implementation |
|---|---|---|---|
| user | net/slirp.c, libslirp | all | libslirp via FFI |
| passt | net/passt.c (10.1) | Linux | spawn passt, stream or vhost-user |
| tap | net/tap.c, tap-linux.c, tap-bsd.c, tap-solaris.c, tap-win32.c | Linux, BSD, Solaris, Windows (TAP-Windows) | native |
| bridge | net/tap.c with qemu-bridge-helper | Linux | native, helper shipped as symlink |
| socket | net/socket.c | all | native |
| stream | net/stream.c (7.2) | all | native |
| dgram | net/dgram.c (7.2) | all | native |
| vde | net/vde.c, libvdeplug | Linux, BSD | libvdeplug via FFI |
| l2tpv3 | net/l2tpv3.c | Linux | native |
| netmap | net/netmap.c | FreeBSD, Linux with netmap | native, ioctl level |
| af-xdp | net/af-xdp.c (8.2), libxdp >= 1.4.0 | Linux | native rings, libxdp only for program load |
| vhost-user | net/vhost-user.c | Linux, BSD | ruvm-vhost (document 13) |
| vhost-vdpa | net/vhost-vdpa.c | Linux | ruvm-vhost (document 13) |
| vmnet-host, vmnet-shared, vmnet-bridged | net/vmnet-*.c (7.1) | macOS | vmnet.framework via objc2 |
| hubport | net/hub.c | all | native |

#### user: libslirp

`-netdev user` is the default network for a VM without configuration, and it is the one most users hit first. QEMU links the external libslirp (the in-tree slirp submodule went away in 7.2; meson still has a `slirp.wrap` fallback and requires libslirp 4.7 only when CFI is enabled). Every `NetdevUserOptions` field (addresses, DHCP and DNS settings, `tftp`, `bootfile`, `smb`, `hostfwd`, `guestfwd`, `restrict`) maps to a `SlirpConfig` field that net/slirp.c fills before `slirp_new()`, and the HMP commands `hostfwd_add` and `hostfwd_remove` map to libslirp calls.

Decision: ruvm links libslirp through a thin `-sys` crate for 1.0 and does not rewrite it. A Rust TCP/IP stack such as smoltcp would differ in many guest-visible details (DHCP and TFTP option handling, SMB share setup, behavior on connection refusal), which the differential tests in document 22 would flag, for little gain on a path nobody runs for speed. The libslirp timer and poll callbacks (`SlirpCb`) are bound to a ruvm-aio reactor; `guestfwd` targets that name a chardev use ruvm-chardev.

#### passt

[passt](https://passt.top/) (Plug A Simple Socket Transport, by Stefano Brivio and David Gibson) is a user-mode network stack that maps guest traffic onto host sockets without NAT tables and without privilege. QEMU 10.1 added a dedicated `passt` netdev in net/passt.c that spawns the daemon itself. With the default datapath QEMU creates a `socketpair()`, hands one end to passt as `--fd`, and exchanges frames with the same 4-byte length-prefixed framing as `-netdev stream`. With `vhost-user=on` QEMU starts passt with `--vhost-user` and connects a vhost-user chardev to it, so frames go directly between passt and the virtio-net rings without crossing QEMU. The roughly 30 fields of `NetdevPasstOptions` (addresses, DNS, port forwards, protocol switches, and a raw `param` list) translate one to one into passt arguments. ruvm builds the same argv, spawns with `posix_spawn` (no glib `GSubprocess`), passes `--pid` to track the child and kills it on netdev removal, and exposes the netdev only on Linux as QEMU does (`CONFIG_PASST`). In ruvm's native CLI (`ruvm run`) the default network for an unprivileged user is passt with vhost-user when a passt binary is found, and user (libslirp) otherwise; the QEMU-compatible binaries keep QEMU's default of `user`.

#### tap and bridge

tap is the workhorse for managed deployments. libvirt passes pre-opened fds (`fd=`, `fds=` with `IFF_MULTI_QUEUE` and one fd per queue pair), which ruvm accepts from the command line, from `getfd` and `add-fd` via QMP, and from SCM_RIGHTS on the monitor socket (document 18). `script` and `downscript` run `/etc/qemu-ifup` and `/etc/qemu-ifdown` by default. `-netdev bridge` and `tap,helper=` run qemu-bridge-helper, a setuid binary that reads /etc/qemu/bridge.conf ACLs, creates a tap, attaches it to the bridge and returns the fd over a socket. ruvm ships `qemu-bridge-helper` as a symlink to the multi-call binary (canon) with the same ACL file format; the helper path is only reachable under that name.

With `vhost=on`, the tap fd is handed to the kernel via `VHOST_NET_SET_BACKEND` and the data path leaves userspace (document 13). Without vhost, ruvm's tap path uses io_uring: TX submits one `IORING_OP_WRITEV` per frame, all frames of a batch in one submission; RX uses a provided buffer ring so the kernel picks buffers and ruvm copies into guest RX buffers. `poll-us` (busy polling time for the vhost-net kernel thread, passed through `VHOST_SET_VRING_BUSYLOOP_TIMEOUT`) is forwarded unchanged. eBPF RSS steering (`set_steering_ebpf`, used by virtio-net's `ebpf-rss-fds`) attaches the program with `TUNSETSTEERINGEBPF`.

On Windows, tap-win32 talks to the TAP-Windows adapter through overlapped I/O; ruvm maps that to IOCP.

#### socket, stream, dgram

`-netdev socket` is the old form: `listen`, `connect` (TCP with a 4-byte big-endian length before each frame), `mcast` (UDP multicast, one frame per datagram), `udp` with `localaddr`. QEMU 7.2 added `stream` (a `SocketAddress`, which can be inet, unix, fd or vsock, with `server` and a client reconnect option, `reconnect-ms` since 9.2) and `dgram` (`local` and `remote` `SocketAddress` pairs, supporting Unix datagram sockets) as replacements that use the QAPI socket address type. passt and socket_vmnet both talk the stream framing. ruvm implements all three over ruvm-aio sockets. For stream the reader keeps a small state machine for partial length prefixes, the same as `SocketReadState` and `net_socket_rs_init()` in net/net.c, and reads with `recv` into a buffer ring so several frames are parsed per wakeup; for dgram it uses `recvmmsg` with 32 messages per call.

#### vde, l2tpv3, netmap

vde links libvdeplug; ruvm binds it via FFI since the VDE plug protocol is defined by that library. l2tpv3 (net/l2tpv3.c) is a static unmanaged L2TPv3 tunnel over raw IP or UDP with optional 32 or 64-bit cookies and counters; QEMU's implementation uses `recvmmsg` with a ring of 64 message headers, and ruvm does the same. netmap (`ifname`, `devname`) maps the netmap rings and uses `NIOCTXSYNC` and `NIOCRXSYNC`; it is mainly used on FreeBSD, and ruvm implements it at the ioctl level without the netmap user library.

#### af-xdp

`-netdev af-xdp` (QEMU 8.2, net/af-xdp.c, contributed by Ilya Maximets) binds AF_XDP sockets to queues of a host NIC. Each socket has four rings shared with the kernel: TX, RX, Fill and Completion. QEMU uses libxdp (1.4.0 or newer) for the UMEM and socket setup and to load a default XDP program that redirects the chosen queues. Unprivileged use goes through `inhibit=on` with `sock-fds` from a privileged manager or, since 10.1, `map-path` to a pinned xsks map. ruvm implements the rings itself on top of `setsockopt(SOL_XDP, ...)` and `mmap` of the ring offsets, and uses libxdp only to load the default program when `inhibit=off`. The UMEM is allocated by ruvm, not in guest memory, so frames are copied once between UMEM and guest buffers; a zero-copy variant that registers guest RAM as UMEM conflicts with memory hotplug and vIOMMU remapping and is an open question for document 25.

#### vmnet

On macOS, QEMU 7.1 added `vmnet-host`, `vmnet-shared` and `vmnet-bridged` (net/vmnet-common.m, vmnet-host.c, vmnet-shared.c, vmnet-bridged.m) using Apple's vmnet.framework. They need macOS 11 or newer and either root or the `com.apple.vm.networking` entitlement. Many macOS users run socket_vmnet instead, a privileged daemon that exposes vmnet over a Unix socket to an unprivileged QEMU through `-netdev stream`. vmnet delivers packets on a dispatch queue; ruvm bridges that to its kqueue reactor with an `EVFILT_USER` trigger and reads with `vmnet_read` in batches (the API takes an array of `vmpktdesc`). ruvm binds the framework through objc2 and block2.

#### vhost-user and vhost-vdpa

`-netdev vhost-user,chardev=...,queues=N` and `-netdev vhost-vdpa,vhostdev=/dev/vhost-vdpa-0` (with `vhostfd`, `queues`, `x-svq`) carry no packets through ruvm; they configure the vhost backend for the virtio-net device. They are specified in document 13, including the shadow virtqueue used for vDPA migration.

### Filters

Netfilters are QOM objects (`-object filter-*,netdev=...`) that sit between a netdev and its peer. Common properties from net/filter.c: `netdev`, `queue` (`all`, `rx`, `tx`), `status` (`on` or `off`, changeable at run time with `qom-set`), `position` (`head`, `tail`, or `id=<filter>`) and `insert` (`before` or `behind`, relative to `position`). Order matters and is visible: QEMU walks filters in list order for TX and in reverse for RX (`netfilter_next()` in net/filter.c), and ruvm keeps that rule.

| Filter | QEMU source | Behavior |
|---|---|---|
| filter-dump | net/dump.c | pcap file writer, `file`, `maxlen` (default 65536 bytes per packet) |
| filter-buffer | net/filter-buffer.c | holds packets and releases them every `interval` microseconds; used by COLO and for latency experiments |
| filter-mirror | net/filter-mirror.c | copies packets to a chardev `outdev`, with `vnet_hdr_support` to send the vnet header length |
| filter-redirector | net/filter-mirror.c | moves packets from the netdev to `outdev` and injects packets read from `indev` |
| filter-rewriter | net/filter-rewriter.c | tracks TCP connections and rewrites sequence numbers on the secondary side for COLO |
| filter-replay | net/filter-replay.c | records and replays network traffic for record/replay (document 17) |
| colo-compare | net/colo-compare.c | compares primary and secondary output packets and releases or triggers checkpoint |

Filter processing in ruvm happens on the netdev's reactor. A filter that holds packets (buffer, colo-compare) takes ownership of the frame buffer instead of copying, and releases through the peer's `RxSink` or `TxBatch` as appropriate. filter-dump writes asynchronously through the reactor's file I/O rather than QEMU's blocking `write` on the main loop, but produces byte-identical pcap output (link type Ethernet, snap length `maxlen`) so pcap-based tests in QEMU's suite compare equal.

COLO (COarse-grained LOck-stepping, docs/system/qemu-colo.rst and docs/colo-proxy.txt; the checkpoint side in migration/colo.c is covered in document 17) uses filter-mirror and filter-redirector on the primary to feed a copy of incoming traffic to the secondary, filter-rewriter on the secondary to reconcile TCP sequence numbers, and colo-compare on the primary with `primary_in`, `secondary_in`, `outdev`, `iothread`, `notify_dev`, `compare_timeout` (default 3000 ms), `expired_scan_cycle` (the check runs every 1000 ms by default), `max_queue_size` (default 1024) and `vnet_hdr_support`. colo-compare runs on an iothread in QEMU and in ruvm. ruvm ports the connection tracking and payload comparison field for field, because the comparison rules decide when a checkpoint fires.

### QMP and events

The netdev QMP surface is `netdev_add`, `netdev_del`, `set_link`, `query-rx-filter` and `announce-self`, plus the events `NIC_RX_FILTER_CHANGED`, `FAILOVER_NEGOTIATED`, `NETDEV_STREAM_CONNECTED`, `NETDEV_STREAM_DISCONNECTED`, and (since 10.0) `NETDEV_VHOST_USER_CONNECTED` and `NETDEV_VHOST_USER_DISCONNECTED`. `announce-self` (net/announce.c) sends gratuitous RARP frames through every NIC, or asks virtio-net guests to announce themselves via `VIRTIO_NET_S_ANNOUNCE`, with `initial`, `max`, `rounds`, `step`, `interfaces` and `id` parameters shared with the post-migration announce. The announce timer state is part of migration for virtio-net; ruvm ports `vmstate_announce_timer` unchanged.

## Character devices

### Model

A chardev (chardev/char.c) is a QOM object of a `TYPE_CHARDEV` subclass with class callbacks for open, write, watches, ioctls, fd passing, client add, disconnect and echo. The frontend side, `CharFrontend` in 11.x (formerly `CharBackend`), registers `can_read`, `read`, `event` and `be_change` handlers with `qemu_chr_fe_set_handlers()`. Flow control is pull based: the backend calls `can_read` to learn how many bytes the frontend accepts (at most `CHR_READ_BUF_LEN`, 4096, per read), then delivers that many. Events are `CHR_EVENT_OPENED`, `CLOSED`, `BREAK`, `MUX_IN`, `MUX_OUT`.

ruvm's `CharBackend` trait keeps those semantics with a reactor instead of a `GMainContext`:

```rust
pub trait CharBackend: Object + Send + Sync {
    fn write(&self, buf: &[u8]) -> Result<usize, CharError>;          // non-blocking, may be partial
    fn write_all_blocking(&self, buf: &[u8]) -> Result<(), CharError>; // qemu_chr_fe_write_all
    fn attach(&self, fe: Arc<dyn CharFrontendOps>, reactor: ReactorId) -> Result<(), CharError>;
    fn detach(&self);
    fn accept_input(&self) {}                                           // frontend has room again
    fn ioctl(&self, req: CharIoctl<'_>) -> Result<(), CharError> { Err(CharError::Unsupported) }
    fn take_msgfds(&self) -> SmallVec<[OwnedFd; 4]> { SmallVec::new() } // SCM_RIGHTS received
    fn set_msgfds(&self, fds: &[BorrowedFd<'_>]) -> Result<(), CharError> { Err(CharError::Unsupported) }
    fn set_echo(&self, echo: bool) {}
    fn set_fe_open(&self, open: bool) {}
    fn disconnect(&self) {}
}
```

`CharIoctl` covers the serial and parallel ioctls (`CHR_IOCTL_SERIAL_SET_PARAMS`, `SET_BREAK`, `GET_TIOCM`, `SET_TIOCM`, `PP_READ_DATA` and friends) that host serial and parallel port passthrough use. Blocking writes exist because QEMU has them: the monitor, some UART models when the guest writes with the FIFO disabled, and `qemu_chr_fe_write_all()` users. In ruvm a blocking write on a vCPU thread is a bounded wait on a writability notification from the reactor, not a spin, and it drops no locks the device does not own. The msgfds pair is how vhost-user and QMP's `getfd` pass file descriptors over Unix sockets.

### Backends

| Backend | QEMU source | Notes |
|---|---|---|
| null | char-null.c | discards output, never produces input |
| file | char-file.c | `path`, `append`, and `input-path` (8.1) for a separate input file |
| pipe | char-pipe.c | `path.in` and `path.out` FIFOs, Windows named pipes |
| serial | char-serial.c | host tty or COM port, termios and TIOCM ioctls; `tty` alias removed in 8.0 |
| parallel | char-parallel.c | Linux ppdev and FreeBSD ppi; `parport` alias removed in 8.0 |
| pty | char-pty.c | allocates a pty, reports path through `query-chardev`, `path` symlink option |
| socket | char-socket.c | TCP, Unix, fd, vsock; server or client; TLS, telnet, tn3270, websocket, reconnect |
| udp | char-udp.c | `host`, `port`, `localaddr`, `localport` |
| stdio | char-stdio.c, char-win-stdio.c | `signal` controls whether Ctrl-C reaches QEMU |
| console | char-console.c | Windows console only |
| ringbuf | char-ringbuf.c | in-memory ring, `size` a power of two; `memory` is a deprecated alias |
| mux | char-mux.c | multiplexes up to 4 frontends, Ctrl-a escape |
| hub | char-hub.c (10.0) | fans in and out to up to 4 backends |
| msmouse | msmouse.c | Microsoft serial mouse protocol from UI input events |
| wctablet | wctablet.c | Wacom serial tablet protocol |
| braille | baum.c | Baum braille display via brlapi |
| testdev | testdev.c | test protocol used by kvm-unit-tests |
| spicevmc, spiceport | chardev/spice.c | SPICE channels (vdagent, usbredir, port channels) |
| qemu-vdagent | ui/vdagent.c | in-process spice-vdagent protocol for clipboard and mouse |
| dbus | ui/dbus-chardev.c | exposes a chardev over the D-Bus display |
| vc | ui/console-vc.c | virtual console with vt100 emulation, `width`, `height`, `cols`, `rows`, and `encoding` (11.1) |

ruvm implements every backend in the `ChardevBackendKind` enum natively in Rust, except braille, which needs brlapi and is linked through FFI behind a build feature, and spicevmc and spiceport, which go through libspice-server (see SPICE below). All of them run on a reactor. File, pipe and serial I/O on Linux use io_uring reads with a single outstanding read per chardev; on macOS they use kqueue readiness; on Windows serial, pipe and console use IOCP with overlapped I/O, replacing QEMU's polling thread for Windows console input.

### socket

The socket chardev is the most used backend after stdio: it carries QMP, HMP, serial consoles, vhost-user control, guest agent channels, and virtio-serial ports. Options: `addr` (or legacy `host`/`port`/`path`/`fd`), `server`, `wait`, `nodelay`, `telnet`, `tn3270`, `websocket`, `tls-creds`, `tls-authz`, `reconnect-ms`, and for Unix sockets `abstract` and `tight`. Behavior details that matter:

- `server=on,wait=on` blocks QEMU startup until a client connects. ruvm blocks machine creation at the same point (after the chardev is created, before the monitor loop starts) so that libvirt's startup sequence is unchanged.
- `telnet=on` sends the same telnet negotiation as `tcp_chr_telnet_init()` (IAC WILL ECHO, IAC WILL SUPPRESS-GO-AHEAD, IAC WILL BINARY, IAC DO BINARY) and strips IAC sequences on input, translating IAC BREAK into `CHR_EVENT_BREAK` for the serial port. `tn3270=on` performs the TN3270 negotiation used by s390x 3270 consoles.
- `websocket=on` wraps the stream in RFC 6455 framing after an HTTP upgrade; the handshake parser is shared with VNC's websocket support, which had a use-after-free in handshake cleanup in QEMU (CVE-2025-11234, fixed in 10.2).
- TLS uses the `tls-creds-x509`, `tls-creds-psk` and `tls-creds-anon` objects (document 19 covers the crypto stack), with `tls-authz` checking the client certificate distinguished name. QEMU 10.2 added multiple x509 identities per credentials object to support parallel certificates with different algorithms; ruvm's TLS layer (rustls, with the same PEM directory layout QEMU reads) supports this from the start.
- `reconnect-ms` makes a client socket retry after disconnect; vhost-user-blk and virtio-net with vhost-user depend on it for backend restart.

### mux and hub

`mux=on` on a chardev (or `-serial mon:stdio`) creates a mux with up to 4 frontends (`MAX_MUX` in chardev/chardev-internal.h), usually a serial port and the HMP monitor. The escape character defaults to Ctrl-a (`-echr` changes it), with Ctrl-a c switching focus, Ctrl-a x quitting, Ctrl-a s saving disks, Ctrl-a b sending break, Ctrl-a t toggling timestamps, Ctrl-a h printing help. Each frontend has a 32-byte input buffer (`MUX_BUFFER_SIZE`). The hub backend added in 10.0 does the opposite: one frontend, up to 4 backends, output copied to all, input merged from all. ruvm ports both with the same buffer sizes, since pasted input is dropped at the same point only if the buffers match.

### vc, the vt100 emulator and text consoles

`vc` chardevs are the text consoles in graphical UIs (the "serial0" and "parallel0" tabs in GTK, and monitor consoles). ui/console-vc.c implements a small vt100 emulator (with parsing in ui/vt100.c) rendering glyphs from ui/vgafont.h into a `DisplaySurface`. QEMU 11.1 added an `encoding` option to vc chardevs (`cp437` or `utf8`), and the vt100 emulator now decodes UTF-8 and renders through the CP437 glyph table (ui/cp437.c), which is also exposed as `org.qemu.Display1.Chardev.VCEncoding` on the D-Bus display. ruvm ports the emulator and font. GTK with VTE uses VTE instead of the internal emulator, as QEMU does when built with VTE.

### Guest-facing agents: qemu-vdagent, spicevmc, dbus

The qemu-vdagent chardev implements the SPICE vdagent protocol inside QEMU, so a guest spice-vdagent on a virtio-serial port can share the clipboard and absolute mouse with any UI (VNC, GTK, D-Bus), not only SPICE. QEMU 10.1 added migration support for it, which means its state (capabilities negotiated, pending clipboard request) is in the migration stream; ruvm ports the VMState. spicevmc and spiceport go through libspice-server. The dbus chardev exports a chardev on the D-Bus display as `org.qemu.Display1.Chardev`, handing a Unix socket fd to the client.

### Monitor attachment

QEMU 11.1 added `-object monitor-qmp,id=...,chardev=...` and `-object monitor-hmp,...` and deprecated `-mon`; `-qmp` and `-monitor` stay as sugar. ruvm accepts all spellings (document 18 owns the monitor) and treats a monitor as one more `CharFrontendOps` on the chardev.

## UI

### Console model

QEMU's display core (ui/console.c, include/ui/console.h) has `QemuConsole` objects: graphic consoles owned by display adapters, and text consoles owned by vc chardevs. A graphic console has a `DisplaySurface` (a pixman image over guest VRAM or a shadow buffer) and optionally an OpenGL scanout (texture or dmabuf). Displays register a `DisplayChangeListener` with 17 ops: surface switch and update, text console updates, cursor and mouse, and the GL scanout texture and dmabuf ops. The refresh timer runs at `GUI_REFRESH_INTERVAL_DEFAULT` (30 ms) and backs off to `GUI_REFRESH_INTERVAL_IDLE` (3000 ms) when nothing changes; adapters without dirty tracking (VGA, some board framebuffers) scan VRAM on each refresh using the memory dirty log.

ruvm's `DisplayListener` trait is the same op set with Rust types: `Surface` (format, stride, and either a shared memory mapping of guest VRAM or an owned buffer), `DirtyRect` lists coalesced per refresh instead of one call per rectangle, `Scanout::Texture` and `Scanout::Dmabuf { planes, modifier, fourcc, y0_top }` (multi-plane dmabufs as QEMU 10.1 added for SPICE and D-Bus), and `Cursor` with ARGB data and hotspot. Display adapters (document 12 for VGA, QXL, ramfb, bochs-display; document 13 for virtio-gpu) produce these; listeners consume them. The console registry keeps QEMU's console indexes, since `-device ...,head=` and QMP `screendump` with `device` and `head` refer to them.

`screendump` writes PPM or PNG (`format` since 7.1) from the current surface or a GL readback. ruvm writes both, byte-compatible for PPM, and PNG through the `png` crate at the same compression level QEMU passes to libpng so image hashes in functional tests match.

### VNC

Decision: ruvm's VNC server is a Rust rewrite, not an FFI binding. QEMU's ui/vnc*.c files are about 11,800 lines of C with its own RFB implementation (it never used libvncserver), and it is one of the most exposed network services in a QEMU deployment. We port the protocol behavior and the encoders, and write the connection handling fresh on ruvm-aio.

Protocol: RFB 3.3, 3.7 and 3.8 handshakes and every `-vnc` option in qemu-options.hx, including `share`, `lossy`, `non-adaptive`, `key-delay-ms`, `audiodev`, `power-control` and `display=`. Unix sockets are supported, and websocket over Unix sockets since 9.1. `change vnc password`, `set_password`, `expire_password`, `display-reload` (reload TLS certificates) and `display-update` (change listen addresses) are QMP entry points; `query-vnc` and `query-vnc-servers` report state in `VncInfo` and `VncInfo2`, including the `VncPrimaryAuth` and `VncVencryptSubAuth` enums.

Encodings QEMU implements, and ruvm implements with identical output for a given input so differential tests can compare framebuffer updates byte for byte:

| Encoding | Number | QEMU source |
|---|---|---|
| Raw | 0 | ui/vnc.c |
| Hextile | 5 | ui/vnc-enc-hextile.c |
| Zlib | 6 | ui/vnc-enc-zlib.c |
| Tight (with JPEG when `lossy` and libjpeg) | 7 | ui/vnc-enc-tight.c |
| Tight PNG | -260 | ui/vnc-enc-tight.c |
| ZRLE | 16 | ui/vnc-enc-zrle.c |
| ZYWRLE | 17 | ui/vnc-enc-zywrle-template.c |

Pseudo-encodings handled: DesktopSize, ExtendedDesktopSize, PointerTypeChange, RichCursor, AlphaCursor, ExtendedKeyEvent (raw scancodes, which is what makes non-US keyboards work without `-k`), QEMU Audio, WMVi (pixel format change), LED state, XVP (power control), extended clipboard, and CompressLevel and QualityLevel ranges. CopyRect and RRE are defined in ui/vnc.h but not sent. Byte-identical Tight output requires using zlib with the same compression levels and strategy per stream and the same JPEG quality table; ruvm links zlib (via libz-sys, not a pure-Rust deflate) and libjpeg-turbo so the compressed bytes match. The framebuffer limit is `VNC_MAX_WIDTH` 5120 by `VNC_MAX_HEIGHT` 2160 with dirty tracking in 16-pixel wide columns (`VNC_DIRTY_PIXELS_PER_BIT`); the update frequency adapts from 30 ms in 50 ms steps up to 3000 ms when the client is idle (`VNC_REFRESH_INTERVAL_*`). The adaptive lossy heuristic over 64 by 64 regions (`VNC_STAT_RECT`) is ported with its thresholds.

Authentication: `VNC_AUTH_NONE` (1), `VNC_AUTH_VNC` (2, DES challenge with the 8-character password limit, kept because clients still use it), `VNC_AUTH_VENCRYPT` (19) with subtypes 256 to 264 (`PLAIN`, `TLSNONE`, `TLSVNC`, `TLSPLAIN`, `X509NONE`, `X509VNC`, `X509PLAIN`, `X509SASL`, `TLSSASL`), and `VNC_AUTH_SASL` (20). The subtype follows from the `tls-creds` type, `password` and `sasl`, and ruvm reproduces the matrix because `query-vnc` reports it. SASL (ui/vnc-auth-sasl.c) uses Cyrus SASL with the `qemu` service name and `/etc/sasl2/qemu.conf`; ruvm links libsasl2 via FFI because GSSAPI and SCRAM mechanisms live there, and supports the SSF layer (SASL data encryption) as QEMU does when TLS is not used.

Threads: QEMU runs one VNC worker thread (ui/vnc-jobs.c) that encodes jobs for all clients, while client sockets are serviced on the main loop. ruvm encodes on a small pool (one task per client per update, capped at the number of host cores divided by four, minimum one), because encoding 5120 by 2160 Tight updates for several clients serialized on one thread is a measurable bottleneck in VDI style deployments. Output order per client is preserved; only independent clients run in parallel.

QEMU 11.1 added `qemu-vnc` (tools/qemu-vnc, docs/tools/qemu-vnc.rst), a standalone VNC server that connects to a running QEMU through `-display dbus` and serves display, input, audio, clipboard and serial chardevs, so the VNC socket and its parsers live outside the QEMU process. Decision: ruvm installs `qemu-vnc` as another symlink to the multi-call binary, with the same options (`--dbus-address`, `--dbus-p2p-fd`, `--bus-name`, `--wait`, `--password`, TLS options), sharing the in-process VNC server code but consuming the D-Bus display interface. It works against both ruvm and QEMU, which is also how we test the D-Bus display for interop.

### SPICE

SPICE (`-spice`, `-display spice-app`, the QXL device in document 12, spicevmc and spiceport chardevs, the spice audiodev) is implemented by libspice-server; QEMU 11.1 requires spice-server 0.15.0 and spice-protocol 0.14.3 or newer. The protocol, image compressors and video streaming all live in that library, with no second implementation. Decision: ruvm links libspice-server via FFI and ports QEMU's glue (ui/spice-core.c, spice-display.c, spice-input.c, spice-app.c, chardev/spice.c, audio/spiceaudio.c, and hw/display/qxl.c's worker interface). libspice-server calls back from its own worker thread; the glue posts into ruvm reactors and never calls device code from the SPICE thread.

The `-spice` options (ports, x509 files, channel TLS policy, SASL, compression and streaming settings, `video-codec`, `gl`, `rendernode`) map one to one onto libspice-server calls. With `gl=on` the guest's GL scanout is passed as a dmabuf to spice-server for a local client over a Unix socket; since QEMU 10.1, remote clients also work with `gl=on` by encoding the GL frames with a video codec. SPICE migration (`client_migrate_info`, the `SPICE_MIGRATE_COMPLETED` event, and the switch-host flow) is forwarded to libspice-server unchanged.

### GTK

QEMU's GTK UI (ui/gtk.c, gtk-egl.c, gtk-gl-area.c, gtk-clipboard.c) is GTK 3 (3.22 or newer), with optional VTE for text consoles. Options: `full-screen`, `window-close`, `show-cursor`, `gl`, and GTK-specific `clipboard`, `grab-on-hover`, `zoom-to-fit`, `show-tabs`, `show-menubar`, `keep-aspect-ratio` and `scale` (the last two new in 10.1). QEMU 11.1 improved console hotplug handling in GTK.

Decision: ruvm builds its GTK frontend on gtk4-rs and GTK 4. The Rust bindings for GTK 3 (gtk3-rs) are archived and unmaintained, with RustSec advisories marking them so (for example RUSTSEC-2024-0415). The frontend reproduces the GTK 3 UI's menus, accelerators (Ctrl-Alt-G grab, Ctrl-Alt-F fullscreen, Ctrl-Alt-number console switch, Ctrl-Alt-plus and minus zoom, Ctrl-Alt-0 fixed zoom, Ctrl-Alt-Q quit) and window title format. GL uses `GtkGLArea` with EGL on both X11 and Wayland, which is what QEMU's gtk-gl-area.c path already does on Wayland. Dmabuf scanouts import through `GdkDmabufTextureBuilder` (GTK 4.14). Text consoles use VTE's GTK 4 build when available, otherwise ruvm's vt100 emulator rendered into the drawing area.

### SDL2

ui/sdl2.c, sdl2-2d.c, sdl2-gl.c, sdl2-input.c: options `gl`, `grab-mod` (`lctrl-lalt`, `lshift-lctrl-lalt`, `rctrl`), `show-cursor`, `window-close`, `full-screen`. ruvm uses the `sdl2` crate over the system SDL2 library, one window per graphic console as QEMU does, with the same hotkeys. SDL3 is not used before 1.0; sdl2-compat covers distributions that retire SDL2.

### Cocoa

ui/cocoa.m: options `left-command-key`, `full-grab`, `swap-opt-cmd`, `zoom-to-fit` (8.2), `zoom-interpolation`, `show-cursor`, `full-screen`. ruvm writes the Cocoa frontend in Rust with objc2 and objc2-app-kit, rendering through a `CALayer` backed view with `IOSurface` when the surface is shared, and keeps QEMU's menu bar.

### curses

`-display curses` (ui/curses.c) shows text-mode VGA contents in a terminal using ncursesw, with `charset` selecting the codepage for the VGA font translation (default CP437). ruvm binds ncursesw via FFI, since the key translation table (ui/curses_keys.h) and wide character behavior are ncurses-specific.

### D-Bus display

`-display dbus` (QEMU 7.0, ui/dbus*.c) exports consoles, keyboard, mouse, multitouch, clipboard, audio and chardevs over D-Bus using the interfaces in ui/dbus-display1.xml, under the `org.qemu.Display1` namespace: `VM`, `Console`, `Keyboard`, `Mouse`, `MultiTouch`, `Listener`, `Clipboard`, `Audio` with `AudioOutListener` and `AudioInListener`, `Chardev` and `Chardev.VCEncoding` (11.1), and the listener extensions `Listener.Unix.Map` (9.2, shared-memory scanout), `Listener.Unix.ScanoutDMABUF2` (multi-plane dmabufs), `Listener.Win32.Map` and `Listener.Win32.D3d11` for Windows hosts. Options: `addr` (bus address), `p2p` (peer-to-peer connections added with QMP `add_client`), `gl`, `rendernode`, `audiodev`. Clients include Marc-André Lureau's GTK 4 based qemu-display project and now QEMU's own qemu-vnc. ruvm implements the server with zbus, generating stubs from the same XML and diffing introspection output against QEMU in CI.

D-Bus is ruvm's recommended path for embedding, since the UI then runs out of process and can crash without taking the VM down.

### egl-headless and the OpenGL paths

GL rendering in QEMU happens for virtio-gpu-gl (virglrenderer), vhost-user-gpu, and virtio-gpu-rutabaga, and GL scanouts reach displays in three ways: as a GL texture in a shared context (`dpy_gl_scanout_texture`), as a dmabuf (`dpy_gl_scanout_dmabuf`), or via readback into a `DisplaySurface`. `-display egl-headless` (ui/egl-headless.c) creates an EGL context on a render node (`rendernode=`), lets virglrenderer render, and reads the result back into a surface so VNC can show a GL guest. ui/egl-helpers.c manages EGL display and context creation (GBM on Linux render nodes, surfaceless platform otherwise), ui/shader.c blits textures, ui/dmabuf.c wraps `QemuDmaBuf` metadata, and ui/udmabuf.c opens /dev/udmabuf so non-GL virtio-gpu blob resources in guest RAM can be exported as dmabufs to GL displays.

ruvm's `ruvm-ui` has a `gl` module that owns EGL through the khronos-egl crate and GBM through FFI, with the same context sharing model: the renderer (virglrenderer in the device's rendering thread, document 13) and the display share a context group, and scanout handoff is a dmabuf fd plus fence where possible. Readback for egl-headless uses a PBO ring of three buffers so the renderer is not stalled by `glReadPixels`. On macOS there is no dmabuf; virtio-gpu-gl on macOS goes through ANGLE or MoltenVK-backed stacks outside ruvm's scope, and the Cocoa UI accepts only surface scanouts in 1.0, as QEMU's Cocoa UI does.

### Input and keymaps

Input events flow from UIs into `QemuInputEvent` (key with `QKeyCode` or raw number, button, absolute and relative axes, multitouch) and to the active input handler (PS/2, USB HID, virtio-input, VNC and SPICE agents), with the routing rules of ui/input.c: a console-bound handler wins, then the most recently activated handler. `input-send-event`, `send-key` and `query-mice` are QMP surfaces. ruvm ports ui/input.c and ui/kbd-state.c (modifier and LED tracking) exactly, because key repeat and modifier edge cases are guest visible.

Keymaps: QEMU generates its keycode translation tables at build time from keycodemapdb (a meson subproject), producing `input-keymap-<from>-to-<to>.c.inc` files for Linux evdev, X11, macOS, Windows, QKeyCode, and XT/AT set 1, 2 and 3 scancodes. ruvm runs the same generator against the same keymaps.csv and emits Rust tables. `-k <layout>` loads reverse keymaps from pc-bios/keymaps (34 layouts), needed only for VNC clients without ExtendedKeyEvent; ruvm ships the same files and the `qemu-keymap` tool (canon binary list) that regenerates them from xkbcommon.

Host input capture: `-object input-linux,evdev=/dev/input/eventN,grab_all=on,repeat=on,grab-toggle=...` (ui/input-linux.c) grabs a host evdev device and feeds the guest directly, and `-object input-barrier` (ui/input-barrier.c) is a client for the Barrier keyboard and mouse sharing protocol. Both are ported natively.

### Clipboard

ui/clipboard.c is a small broker: peers register a `QemuClipboardPeer` and publish `QemuClipboardInfo` for a selection (`CLIPBOARD`, `PRIMARY`, `SECONDARY`) with data types, of which only `QEMU_CLIPBOARD_TYPE_TEXT` (UTF-8 plain text) exists. Peers are qemu-vdagent (the guest side), VNC extended clipboard (ui/vnc-clipboard.c), GTK, Cocoa and D-Bus. Requests are asynchronous: a peer announces ownership, another peer requests data, the owner responds. ruvm implements the broker as a single actor on the main reactor with the same state machine, including the serial number rule vdagent uses to resolve simultaneous grabs, and keeps the one-type limitation; adding image types would change what the vdagent peer advertises to the guest.

## Audio

### Model

`-audiodev <driver>,id=...` creates an audio backend; sound devices (document 12 for AC97, ES1370, SB16, HDA, USB audio, PC speaker, and board codecs; document 13 for virtio-sound) take an `audiodev=` property. Since 8.2 a device without `audiodev=` gets no implicit backend when `-nodefaults` is given; `-audio driver[,model=...]` (7.1) is shorthand that creates both. Every audiodev has `in` and `out` per-direction options: `mixing-engine` (default on), `fixed-settings` (default on), `frequency` (44100), `channels` (2), `format` (s16), `voices` (1 with mixing engine), `buffer-length`, plus the global `timer-period` (default 10000 microseconds). The validation rules in audio/audio.c (for example "You can't use fixed-settings without mixeng") and their error strings are reproduced.

In QEMU 11.x the backend is a QOM class, `TYPE_AUDIO_BACKEND` with `AudioBackendClass` callbacks (`realize`, `open_out`, `open_in`, `close_*`, `is_active_*`, `set_active_*`, `set_volume_*`, `write`, `read`, and more) in include/qemu/audio.h, with audio/audio-be.c and audio/audio-mixeng-be.c. Underneath, each driver provides hardware voices (`HWVoiceOut`, `HWVoiceIn`), and each guest device opens software voices (`SWVoiceOut`, `SWVoiceIn`) that the mixing engine mixes into or splits from hardware voices.

ruvm's `AudioBackend` trait is the hardware-voice side (open, start, stop, a buffer to write into or read from, a latency query, volume); the mixing engine and software voices are generic code in ruvm-audio. The audio timer, which QEMU runs on the main loop, runs on a dedicated audio thread in ruvm so a busy main thread cannot starve playback. Callback-driven backends (PipeWire, JACK, CoreAudio, SDL, PulseAudio in its threaded mainloop) pull from a lock-free SPSC ring that the audio thread fills.

### The mixing engine

audio/mixeng.c and its templates convert every supported format (u8, s8, u16, s16, u32, s32, f32, in either endianness, mono or stereo) into an internal `struct st_sample` of two `int64_t` values (`l`, `r`; a float variant exists under `FLOAT_MIXENG`), mix software voices by adding into that buffer, apply `mixeng_volume` per voice, clip back to the hardware format with saturating conversion, and resample with the rate converter in audio/rate_template.h. The converter keeps a 32.32 fixed-point output position (`opos`, incremented by `(inrate << 32) / outrate`), an integer input position, and the last input sample, and linearly interpolates `(ilast * (UINT_MAX - t) + icur * t) >> 32` between samples; when rates are equal it copies.

Decision: ruvm ports the mixing engine bit for bit, including the 32.32 rate converter, clipping, the volume law and the integer sample type. The reason is testability and migration, not audio quality: our differential tests compare wav audiodev output against QEMU, and a VM migrated between QEMU and ruvm mid-playback should not change what the host hears. Within that constraint the implementation is vectorized: conversion and mixing loops are written with portable SIMD (`std::simd` is not stable, so hand-written `core::arch` paths for x86-64 SSE2 and AVX2 and aarch64 NEON), with the scalar port as the oracle in tests. `mixing-engine=off` passes device PCM straight to the backend in the device's format, as QEMU does, which is the low-latency path for single-voice guests.

A better resampler (windowed sinc) is useful for audio quality with `fixed-settings=on` at 44100 Hz and guests playing 48000 Hz. It is available as `x-resampler=sinc` on the audiodev, off by default, reported for document 25.

### Backends

| Driver | QEMU source | Hosts | ruvm binding | Notes |
|---|---|---|---|---|
| none | audio/noaudio.c | all | native | consumes at real-time rate |
| wav | audio/wavaudio.c | all | native | `path`, writes WAV header on close |
| alsa | audio/alsaaudio.c | Linux | alsa crate over alsa-lib | `try-poll` default false since 10.1 |
| oss | audio/ossaudio.c | Linux, BSD | native ioctls | |
| pa | audio/paaudio.c | Linux, BSD | libpulse via FFI | |
| pipewire | audio/pwaudio.c (8.1) | Linux | libpipewire via pipewire-rs | libpipewire 0.3.60 or newer |
| jack | audio/jackaudio.c | Linux, macOS, BSD | libjack via FFI | |
| sndio | audio/sndioaudio.c | OpenBSD, others | libsndio via FFI | `dev`, `latency` |
| coreaudio | audio/coreaudio.m | macOS | objc2-core-audio | `buffer-count` |
| dsound | audio/dsoundaudio.c | Windows | windows crate | `latency` |
| sdl | audio/sdlaudio.c | all with SDL2 | sdl2 crate | `buffer-count` |
| spice | audio/spiceaudio.c | with SPICE | libspice-server | playback and record channels |
| dbus | audio/dbusaudio.c | with D-Bus display | zbus | `nsamples` (10.0) |

Option names, defaults and the per-direction structure follow qapi/audio.json exactly, since libvirt generates them. QEMU 10.1 added float endianness converters used by ALSA and changed the ALSA `try-poll` default to false; ruvm matches the 11.1 defaults. The HMP `wavcapture` and `stopcapture` commands are deprecated since 10.2 in favor of `-audiodev wav` or host capture; ruvm implements them for compatibility (audio/wavcapture.c semantics) and marks them deprecated in `help` output identically.

VNC audio uses the audiodev named in `-vnc ...,audiodev=`; D-Bus audio is what qemu-vnc forwards to VNC clients.

### Latency

Playback latency with the mixing engine is the device's buffering plus one `timer-period` plus the backend buffer; with the 10 ms timer and a 1024-frame PipeWire quantum at 48 kHz the floor is above 30 ms before guest buffering. ruvm keeps the defaults, but the dedicated audio thread allows `timer-period` values down to 1000 microseconds without main loop impact, and callback backends can be driven by the backend's own period instead of the timer when `x-backend-clock=on`. Both are measured in document 21.

## Testing

The QEMU qtests that cover these subsystems run unchanged through the qtest accelerator (document 22): net filter tests (test-filter-mirror, test-filter-redirector, test-netfilter), test-filter-buffer, netdev-socket, chardev unit tests ported from tests/unit/test-char.c, vnc-display-test, dbus-display-test, dbus-vnc-test, and virtio-net tests with every backend available on the CI host. Differential tests against QEMU 11.1 compare filter-dump pcaps, VNC update bytes for every encoding, `info network` and `query-chardev` output, `screendump` output, and wav audiodev output for a fixed PCM stream. Fuzzing targets: the VNC client message parser (pre-auth and post-auth), the websocket handshake, the telnet and tn3270 parsers in the socket chardev, colo-compare's packet parser, and the vdagent protocol parser.

## Decisions and open items

- libslirp stays via FFI for `-netdev user` in 1.0; a Rust user-mode stack is not planned.
- `ruvm run` defaults to passt with vhost-user when passt is installed; QEMU-compatible binaries keep `user`.
- VNC is a Rust rewrite with byte-identical encoder output (linking zlib and libjpeg-turbo) and parallel per-client encoding.
- `qemu-vnc` is installed as a multi-call symlink.
- SPICE, libsasl2, brlapi, libvdeplug, ncursesw, libpulse, libjack, libsndio are FFI dependencies behind build features.
- GTK frontend on gtk4-rs (GTK 4), not GTK 3; the menu and hotkeys match QEMU's GTK 3 UI.
- D-Bus display on zbus, generated from ui/dbus-display1.xml.
- Mixing engine ported bit for bit with SIMD, `x-resampler=sinc` and `x-backend-clock=on` as off-by-default extensions.
- Audio timer on a dedicated thread; GTK on a dedicated UI thread; Cocoa on the process main thread.
- Open: AF_XDP zero-copy with guest RAM as UMEM, and whether to add more clipboard types if QEMU does.
