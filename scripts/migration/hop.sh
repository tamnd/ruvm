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
#   POSTCOPY  1 to switch each hop to postcopy with migrate-start-postcopy right after it
#             starts, with precopy held to 1 MiB/s so that the switch comes first
#   MULTIFD   1 to migrate with the multifd capability; the destination then starts with
#             "-incoming defer" and gets the capability before migrate-incoming
#   MULTIFD_CHANNELS     multifd-channels (default 2)
#   MULTIFD_COMPRESSION  multifd-compression, "none" or "zlib" (default none)
#   XBZRLE    1 to migrate with the xbzrle capability; the source shows its query-migrate,
#             with the XBZRLE counters, once the hop completed
#   XBZRLE_CACHE_SIZE    xbzrle-cache-size in bytes (default 67108864)
#   FILE      1 to migrate through a file, $WORK/hopN.mig, instead of TCP: the source migrates
#             into it and quits, and only then does the destination start, with
#             "-incoming defer", and load it with migrate-incoming
#   MAPPED_RAM  1 to turn the mapped-ram capability on, for FILE=1
#   DIRECT_IO   1 to set the direct-io parameter, which takes effect with MAPPED_RAM=1 and
#               MULTIFD=1
#   SNAPSHOT  1 to move the guest with savevm and loadvm instead of a migration: the source
#             adds the qcow2 image $WORK/snap.qcow2 with blockdev-add, saves the snapshot
#             "hopN" into it and quits, then the destination starts with -S, adds the image,
#             loads the snapshot and continues. Both list the snapshots with "info snapshots".
#             The raw disks of the io guest cannot hold snapshots, so this is for the others
#   QEMU_IMG  the qemu-img that creates the image for SNAPSHOT (default: qemu-img on PATH)
#   BG_SNAPSHOT  1 to save into the file of FILE=1 (which it turns on) with the
#             background-snapshot capability: the guest goes on while RAM is saved as it was
#             when the snapshot started. The source runs past that point, so instead of
#             joining the outputs each destination is checked on its own, without its first
#             line, which may be partial, and the line it then starts at has to be one the
#             source printed too
#   CPR       "reboot" to migrate with the cpr-reboot mode through the file of FILE=1 (which
#             it turns on). "transfer" for cpr-transfer: both run with their RAM in a shared
#             memory backend, the source sends its descriptor over the UNIX socket
#             $WORK/cprN.sock of the "cpr" channel and then the rest over TCP, and the
#             destination, which takes the descriptor before it builds the machine and so only
#             answers on QMP after the source migrated, maps the very same RAM, which then is
#             not sent at all
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
    pids[n]=$!
}

[ "${BG_SNAPSHOT:-0}" = 1 ] && FILE=1
CPR=${CPR:-}
[ "$CPR" = reboot ] && FILE=1
if [ "$CPR" = transfer ]; then
    [ "$guest" = io ] && { echo "CPR=transfer is for the checksum and irq guests" >&2; exit 2; }
    M+=(-object memory-backend-ram,id=pc.ram,size=128M,share=on
        -machine memory-backend=pc.ram,aux-ram-share=on)
fi
last=$((${#kinds[@]} - 1))
rm -f vm*.txt vm*.sock vm*.err
pids=()
if [ "${SNAPSHOT:-0}" = 1 ]; then
    rm -f snap.qcow2
    "${QEMU_IMG:-qemu-img}" create -f qcow2 snap.qcow2 1G >/dev/null || exit 1
    snapadd='{"execute":"blockdev-add","arguments":{"driver":"qcow2","node-name":"snap0","file":{"driver":"file","filename":"'"$WORK/snap.qcow2"'"}}}'
fi
start 0
outputs=(vm0.txt)
for n in $(seq 1 "$last"); do
    port=$((PORT + n - 1))
    caps='{"capability":"events","state":true}'
    bw=4294967296
    post=()
    if [ "${POSTCOPY:-0}" = 1 ]; then
        caps+=',{"capability":"postcopy-ram","state":true}'
        bw=1048576
        post=('{"execute":"migrate-start-postcopy"}')
    fi
    params='"max-bandwidth":'"$bw"',"downtime-limit":'"${DOWNTIME:-300}"
    uri="tcp:127.0.0.1:$port"
    if [ "${FILE:-0}" = 1 ]; then
        uri="file:$WORK/hop$n.mig"
        rm -f "$WORK/hop$n.mig"
    fi
    if [ "${MAPPED_RAM:-0}" = 1 ]; then
        caps+=',{"capability":"mapped-ram","state":true}'
    fi
    if [ "${BG_SNAPSHOT:-0}" = 1 ]; then
        caps+=',{"capability":"background-snapshot","state":true}'
    fi
    if [ -n "$CPR" ]; then
        params+=',"mode":"cpr-'"$CPR"'"'
    fi
    if [ "${DIRECT_IO:-0}" = 1 ]; then
        params+=',"direct-io":true'
    fi
    if [ "${MULTIFD:-0}" = 1 ]; then
        caps+=',{"capability":"multifd","state":true}'
        params+=',"multifd-channels":'"${MULTIFD_CHANNELS:-2}"',"multifd-compression":"'"${MULTIFD_COMPRESSION:-none}"'"'
    fi
    srcinfo=()
    if [ "${XBZRLE:-0}" = 1 ]; then
        caps+=',{"capability":"xbzrle","state":true}'
        params+=',"xbzrle-cache-size":'"${XBZRLE_CACHE_SIZE:-67108864}"
        srcinfo=('{"execute":"query-migrate"}')
    fi
    setcaps='{"execute":"migrate-set-capabilities","arguments":{"capabilities":['"$caps"']}}'
    setparams='{"execute":"migrate-set-parameters","arguments":{'"$params"'}}'
    if [ "${SNAPSHOT:-0}" = 1 ]; then
        sleep "${WARM:-15}"
        echo "--- hop $n: ${kinds[$((n - 1))]} to ${kinds[$n]} through savevm hop$n"
        # Only the first one has to add the image; the others loaded from it.
        add=()
        [ "$n" = 1 ] && add=("$snapadd")
        qmp "vm$((n - 1)).sock" "${add[@]}" "hmp:savevm hop$n" "hmp:info snapshots" \
            '{"execute":"query-status"}' '{"execute":"quit"}'
        # The image stays locked until the source is gone.
        wait "${pids[$((n - 1))]}"
        start "$n" -S
        qmp "vm$n.sock" "$snapadd" "hmp:loadvm hop$n" '{"execute":"query-status"}' \
            '{"execute":"cont"}' '{"execute":"query-status"}'
        outputs+=("vm$n.txt")
        continue
    fi
    if [ "$CPR" = transfer ]; then
        cpr="$WORK/cpr$n.sock"
        rm -f "$cpr"
        # The destination waits for the descriptors before it even answers on QMP.
        start "$n" -incoming "$uri" \
            -incoming '{"channel-type":"cpr","addr":{"transport":"socket","type":"unix","path":"'"$cpr"'"}}'
        sleep "${WARM:-15}"
        echo "--- hop $n: ${kinds[$((n - 1))]} to ${kinds[$n]} with cpr-transfer over $cpr"
        channels='[{"channel-type":"main","addr":{"transport":"socket","type":"inet","host":"127.0.0.1","port":"'"$port"'"}},{"channel-type":"cpr","addr":{"transport":"socket","type":"unix","path":"'"$cpr"'"}}]'
        qmp "vm$((n - 1)).sock" "$setcaps" "$setparams" \
            '{"execute":"migrate","arguments":{"channels":'"$channels"'}}' 'wait:completed' \
            '{"execute":"query-migrate"}' '{"execute":"query-status"}' '{"execute":"quit"}'
        qmp "vm$n.sock" '{"execute":"query-migrate"}' '{"execute":"query-status"}'
        outputs+=("vm$n.txt")
        continue
    fi
    if [ "${FILE:-0}" = 1 ]; then
        # The file has to be whole before anything reads it.
        sleep "${WARM:-15}"
        echo "--- hop $n: ${kinds[$((n - 1))]} to ${kinds[$n]} through $uri"
        qmp "vm$((n - 1)).sock" "$setcaps" "$setparams" \
            '{"execute":"migrate","arguments":{"uri":"'"$uri"'"}}' 'wait:completed' \
            '{"execute":"query-migrate"}' '{"execute":"query-status"}' '{"execute":"quit"}'
        ls -l "$WORK/hop$n.mig"
        start "$n" -incoming defer
        qmp "vm$n.sock" "$setcaps" "$setparams" \
            '{"execute":"migrate-incoming","arguments":{"uri":"'"$uri"'"}}' 'wait:completed' \
            '{"execute":"query-status"}'
        rm -f "$WORK/hop$n.mig"
        outputs+=("vm$n.txt")
        continue
    fi
    if [ "${MULTIFD:-0}" = 1 ]; then
        # Multifd has to be on before the destination listens.
        start "$n" -incoming defer
        qmp "vm$n.sock" '{"execute":"query-status"}' "$setcaps" "$setparams" \
            '{"execute":"migrate-incoming","arguments":{"uri":"'"$uri"'"}}'
    else
        start "$n" -incoming "$uri"
        qmp "vm$n.sock" '{"execute":"query-status"}' "$setcaps"
    fi
    sleep "${WARM:-15}"
    echo "--- hop $n: ${kinds[$((n - 1))]} to ${kinds[$n]}"
    qmp "vm$((n - 1)).sock" "$setcaps" "$setparams" \
        '{"execute":"migrate","arguments":{"uri":"'"$uri"'"}}' "${post[@]}" 'wait:completed' \
        "${srcinfo[@]}" '{"execute":"query-status"}' '{"execute":"quit"}'
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
if [ "${BG_SNAPSHOT:-0}" = 1 ]; then
    for n in $(seq 1 "$last"); do
        echo "--- ${kinds[$n]} ($n) without its first line"
        tail -n +2 "vm$n.txt" >"vm$n.whole"
        "${check[@]}" "vm$n.whole" || status=1
        # The first field counts the lines; the rest of an irq line depends on timing.
        first=$(awk 'NR == 1 { print $1 }' "vm$n.whole")
        if [ -n "$first" ] && grep -q "^$first " "vm$((n - 1)).txt"; then
            echo "resumes at line $first, which ${kinds[$((n - 1))]} ($((n - 1))) printed with $(sed -n "/^$first /,\$p" "vm$((n - 1)).txt" | tail -n +2 | wc -l) more lines after it"
        else
            echo "${kinds[$n]} ($n) resumes at line $first, which ${kinds[$((n - 1))]} ($((n - 1))) never printed"
            status=1
        fi
    done
    exit "$status"
fi
echo "--- all outputs, joined"
"${check[@]}" "${outputs[@]}"
