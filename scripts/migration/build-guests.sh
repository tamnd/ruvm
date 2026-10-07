#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
#
# usage: build-guests.sh OUTDIR
#
# Assembles the migration test guests into OUTDIR/checksum-guest.bin and OUTDIR/irq-guest.bin,
# 64 KiB -bios images. Needs binutils for x86; set CROSS (for example x86_64-linux-gnu-) when
# the host's are for another architecture.
set -e
out=${1:?usage: build-guests.sh OUTDIR}
src=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
for g in checksum-guest irq-guest; do
    "${CROSS}as" --32 -o "$out/$g.o" "$src/$g.S"
    "${CROSS}ld" -m elf_i386 -Ttext=0xffff0000 -e 0xffff0000 -o "$out/$g.elf" "$out/$g.o"
    "${CROSS}objcopy" -O binary "$out/$g.elf" "$out/$g.bin"
    rm -f "$out/$g.o" "$out/$g.elf"
done
echo "built $out/checksum-guest.bin $out/irq-guest.bin"
