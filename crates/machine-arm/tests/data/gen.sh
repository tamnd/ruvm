#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Regenerates the reference data of tests/virt.rs with QEMU 11.1 (qemu-system-aarch64 and dtc
# on PATH; clang and an ld.lld for the hello test, see build_hello below). Run it from this
# directory. Every QEMU blob is the full buffer `-M virt,dumpdtb=` writes, gzipped.
set -euo pipefail

QEMU=${QEMU:-qemu-system-aarch64}
QEMU_SRC=${QEMU_SRC:-$HOME/src/qemu-v11.1.0}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# A fake arm64 Image: the 64 byte header with text_offset 0 and image_size 0x20000, padded to
# 4 KiB. QEMU never runs it with dumpdtb.
python3 - "$TMP/Image" <<'PY'
import struct, sys
hdr = struct.pack('<IIQQQQQQ4sI', 0x14000000, 0, 0, 0x20000, 0, 0, 0, 0, b'ARM\x64', 0)
open(sys.argv[1], 'wb').write(hdr + bytes(4096 - len(hdr)))
PY
head -c 1000 /dev/zero | tr '\0' 'r' > "$TMP/initrd"

M=virt,gic-version=3,its=off,dtb-randomness=off
dump() {
    local name=$1
    shift
    "$QEMU" -nodefaults -display none -M "$M,dumpdtb=$TMP/$name.dtb" "$@"
    gzip -9n -c "$TMP/$name.dtb" > "$name.dtb.gz"
}

dump virt-a57 -cpu cortex-a57,pmu=off
dump virt-a57-smp2-linux -cpu cortex-a57,pmu=off -smp 2 -m 512M \
    -kernel "$TMP/Image" -initrd "$TMP/initrd" -append "console=ttyAMA0 root=/dev/vda"
dump virt-max-smp3 -cpu max,pmu=off -smp 3 -m 1G
dump virt-a76-smp20 -cpu cortex-a76,pmu=off -smp 20 -m 256M

dump_m() {
    local name=$1 opts=$2
    shift 2
    "$QEMU" -nodefaults -display none -M "$M$opts,dumpdtb=$TMP/$name.dtb" "$@"
    gzip -9n -c "$TMP/$name.dtb" > "$name.dtb.gz"
}
head -c 4096 /dev/zero > "$TMP/bios.fd"
dump_m virt-max-el2-el3 ,virtualization=on,secure=on -cpu max,pmu=off -smp 2
dump_m virt-a57-el2-serial2 ,virtualization=on -cpu cortex-a57,pmu=off -smp 2 \
    -serial null -serial null
dump_m virt-max-secure-bios ,secure=on -cpu max,pmu=off -smp 2 -bios "$TMP/bios.fd"

# The default msi=auto, which is the ITS, and a second redistributor region past 123 CPUs.
dump_its() {
    local name=$1
    shift
    "$QEMU" -nodefaults -display none -M "virt,gic-version=3,dtb-randomness=off,dumpdtb=$TMP/$name.dtb" "$@"
    gzip -9n -c "$TMP/$name.dtb" > "$name.dtb.gz"
}
dump_its virt-a57-smp2-its -cpu cortex-a57,pmu=off -smp 2
dump_its virt-a57-smp130-its -cpu cortex-a57,pmu=off -smp 130

# iommu=smmuv3: the SMMUv3 node and the iommu-map of the PCIe node, and the same with the
# root bus bypassing it, which drops the iommu-map.
dump_its virt-a57-smmuv3 -cpu cortex-a57,pmu=off -M iommu=smmuv3
dump_its virt-a57-smmuv3-bypass -cpu cortex-a57,pmu=off \
    -M iommu=smmuv3,default-bus-bypass-iommu=on

# A user -dtb: QEMU drops its memory nodes and /psci and adds its own.
dtc -q -I dts -O dtb -o "$TMP/user.dtb" user.dts
cp "$TMP/user.dtb" user.dtb
dump virt-user-dtb -cpu cortex-a57,pmu=off -m 256M -kernel "$TMP/Image" -append "quiet" \
    -dtb "$TMP/user.dtb"

build_hello() {
    local q=$QEMU_SRC/tests/tcg cc=${CC:-clang} lld=${LLD:-ld.lld}
    local f="--target=aarch64-none-elf -nostdlib -ffreestanding -fno-stack-protector -O0"
    f="$f -isystem $q/minilib"
    $cc $f -x assembler-with-cpp -c "$q/aarch64/system/boot.S" -o "$TMP/boot.o"
    $cc $f -c "$q/minilib/printf.c" -o "$TMP/printf.o"
    $cc $f -c "$q/multiarch/system/hello.c" -o "$TMP/hello.o"
    $lld -static -T "$q/aarch64/system/kernel.ld" "$TMP/hello.o" "$TMP/printf.o" \
        "$TMP/boot.o" -o "$TMP/hello"
    gzip -9n -c "$TMP/hello" > hello.gz
}
if [ "${BUILD_HELLO:-0}" = 1 ]; then
    build_hello
fi
