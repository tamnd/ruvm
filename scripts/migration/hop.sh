#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# usage: hop.sh KIND KIND [KIND...] [GUEST]
#
# Live migrates a test guest over TCP along a chain of emulators, each KIND "qemu" or "ruvm":
# "hop.sh qemu ruvm" is one hop, "hop.sh qemu ruvm qemu irq" goes there and back. It checks
# that the guest's output carries on across every hop. GUEST is "checksum" (checksum-guest.S,
# the default), "irq" (irq-guest.S) or "io" (io-guest.c, a Linux guest doing disk, network and
# rng I/O on virtio-blk, virtio-scsi, two virtio-net on one hub and virtio-rng, with a
# virtio-balloon and a virtio-serial console plugged too; build-io-guest.sh WORK makes it).
#
# Environment:
#   QEMU      qemu-system-x86_64 to run for "qemu" (default: qemu-system-x86_64 on PATH)
#   RUVM      the ruvm binary to run for "ruvm" (default: ruvm on PATH)
#   WORK      scratch directory (default /tmp/ruvm-hop); the guests are built there if missing
#   MACHINE   the -M option (default q35)
#   PORT      first TCP port (default 4444); hop N uses PORT + N
#   WARM      seconds each emulator runs before it migrates (default 15, 120 for io)
#   DATADIR   a -L firmware directory for both, which io needs when ruvm has none installed
#   TSC_KHZ   for io on microvm, which has no timer to calibrate the TSC against under TCG:
#             the tsc_early_khz= to boot with (default: the host's, from /proc/cpuinfo)
#   RUN       seconds the last one runs after the last hop (default 20)
#   DOWNTIME  downtime-limit in ms (default 300)
set -u
here=$(cd "$(dirname "$0")" && pwd)
usage="usage: hop.sh KIND KIND [KIND...] [GUEST]"
kinds=()
guest=checksum
for a in "$@"; do
    case $a in
    qemu | ruvm) kinds+=("$a") ;;
    checksum | irq | io) guest=$a ;;
    *) echo "$usage" >&2; exit 2 ;;
    esac
done
[ ${#kinds[@]} -ge 2 ] || { echo "$usage" >&2; exit 2; }
WORK=${WORK:-/tmp/ruvm-hop}
mkdir -p "$WORK/bin"
cd "$WORK" || exit 1
if [ "$guest" = io ]; then
    [ -f io-guest.cpio.gz ] || "$here/build-io-guest.sh" "$WORK" >/dev/null || exit 1
else
    [ -f "$guest-guest.bin" ] || "$here/build-guests.sh" "$WORK" >/dev/null || exit 1
fi
# ruvm picks its personality from argv[0].
ln -sf "$(command -v "${RUVM:-ruvm}")" "$WORK/bin/qemu-system-x86_64"
bin() {
    case $1 in
    qemu) command -v "${QEMU:-qemu-system-x86_64}" ;;
    ruvm) echo "$WORK/bin/qemu-system-x86_64" ;;
    esac
}
PORT=${PORT:-4444}
MACHINE=${MACHINE:-q35}
M=(-M "$MACHINE" -nodefaults -accel tcg -display none)
[ -n "${DATADIR:-}" ] && M+=(-L "$DATADIR")
check=(python3 "$here/expect.py" "$guest")
if [ "$guest" = io ]; then
    check=(python3 "$here/io-expect.py")
    WARM=${WARM:-120}
    append="console=ttyS0 quiet panic=-1"
    if [ "$MACHINE" = microvm ]; then
        v=device
        khz=${TSC_KHZ:-$(awk -F': ' '/^cpu MHz/ { printf "%d", $2 * 1000; exit }' /proc/cpuinfo)}
        append="$append tsc_early_khz=$khz tsc=reliable"
    else
        v=pci
    fi
    # shellcheck disable=SC2054 # the commas are in -drive and -device options
    M+=(-m 512M -kernel io-guest-vmlinuz -initrd io-guest.cpio.gz -append "$append"
        -drive file=io-guest-disk.img,format=raw,if=none,id=d0 -device "virtio-blk-$v,drive=d0"
        -drive file=io-guest-disk2.img,format=raw,if=none,id=d1
        -device "virtio-scsi-$v,id=scsi0" -device scsi-hd,drive=d1,bus=scsi0.0
        -netdev hubport,id=n0,hubid=0 -netdev hubport,id=n1,hubid=0
        -device "virtio-net-$v,netdev=n0,mac=52:54:00:12:34:01"
        -device "virtio-net-$v,netdev=n1,mac=52:54:00:12:34:02"
        -device "virtio-rng-$v" -device "virtio-balloon-$v"
        -device "virtio-serial-$v,max_ports=1" -device virtconsole)
else
    M+=(-m 128M -bios "$guest-guest.bin")
fi
out() {
    if [ "$guest" != checksum ]; then
        echo "-serial file:$1"
    else
        echo "-chardev file,id=d,path=$1 -device isa-debugcon,iobase=0xe9,chardev=d"
    fi
}
qmp() { python3 "$here/qmp.py" "$@"; }
# start N [ARGS...]: starts emulator N of the chain with its output in vmN.txt.
start() {
    local n=$1
    shift
    rm -f "vm$n.txt" "vm$n.sock" "vm$n.err"
    # shellcheck disable=SC2046
    "$(bin "${kinds[$n]}")" "${M[@]}" $(out "vm$n.txt") -qmp "unix:vm$n.sock,server=on,wait=off" "$@" \
        2>"vm$n.err" &
}

last=$((${#kinds[@]} - 1))
rm -f vm*.txt vm*.sock vm*.err
start 0
outputs=(vm0.txt)
for n in $(seq 1 "$last"); do
    port=$((PORT + n - 1))
    start "$n" -incoming "tcp:127.0.0.1:$port"
    qmp "vm$n.sock" '{"execute":"query-status"}'
    sleep "${WARM:-15}"
    echo "--- hop $n: ${kinds[$((n - 1))]} to ${kinds[$n]}"
    qmp "vm$((n - 1)).sock" \
        '{"execute":"migrate-set-capabilities","arguments":{"capabilities":[{"capability":"events","state":true}]}}' \
        '{"execute":"migrate-set-parameters","arguments":{"max-bandwidth":4294967296,"downtime-limit":'"${DOWNTIME:-300}"'}}' \
        '{"execute":"migrate","arguments":{"uri":"tcp:127.0.0.1:'"$port"'"}}' 'wait:completed' \
        '{"execute":"query-status"}' '{"execute":"quit"}'
    qmp "vm$n.sock" '{"execute":"query-migrate"}' '{"execute":"query-status"}'
    outputs+=("vm$n.txt")
done
echo "--- last: ${kinds[$last]}"
qmp "vm$last.sock" "sleep:${RUN:-20}" '{"execute":"query-status"}' '{"execute":"quit"}'
wait
status=0
for n in $(seq 0 "$last"); do
    echo "--- output of ${kinds[$n]} ($n)"
    "${check[@]}" "vm$n.txt" | tail -1
    if [ "$n" -gt 0 ] && [ "$(wc -l <"vm$n.txt" 2>/dev/null || echo 0)" -lt 2 ]; then
        echo "${kinds[$n]} ($n) printed no full line after the hop"
        status=1
    fi
done
echo "--- stderr"
cat vm*.err
[ "$status" = 0 ] || exit 1
echo "--- all outputs, joined"
"${check[@]}" "${outputs[@]}"
