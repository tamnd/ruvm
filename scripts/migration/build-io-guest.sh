#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
#
# usage: build-io-guest.sh OUTDIR [KERNEL]
#
# Assembles the virtio workload guest into OUTDIR: io-guest.cpio.gz, an initramfs holding
# io-guest.c built static as /init, io-guest-vmlinuz, a copy of KERNEL (by default the running
# one, which needs virtio pci, mmio, blk, net and scsi built in), and io-guest-disk.img and
# io-guest-disk2.img, two zeroed 64 MiB disks made only when missing. The initramfs also gets
# virtio-rng.ko from /lib/modules when KERNEL is named vmlinuz-RELEASE and the driver is a
# module there. Needs gcc, a static libc and cpio.
set -e
out=${1:?usage: build-io-guest.sh OUTDIR [KERNEL]}
kernel=${2:-/boot/vmlinuz-$(uname -r)}
src=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/root/dev" "$tmp/root/proc" "$tmp/root/sys"
gcc -static -O2 -Wall -o "$tmp/root/init" "$src/io-guest.c"
release=$(basename "$kernel" | sed -n 's/^vmlinuz-//p')
for m in /lib/modules/"$release"/kernel/drivers/char/hw_random/virtio-rng.ko*; do
    if [ -n "$release" ] && [ -f "$m" ]; then
        cp "$m" "$tmp/root/"
    fi
done
(cd "$tmp/root" && find . | cpio -o -H newc --quiet) | gzip -9 >"$out/io-guest.cpio.gz"
cp "$kernel" "$out/io-guest-vmlinuz"
for d in io-guest-disk.img io-guest-disk2.img; do
    [ -e "$out/$d" ] || dd if=/dev/zero of="$out/$d" bs=1M count=64 status=none
done
echo "built $out/io-guest.cpio.gz $out/io-guest-vmlinuz $out/io-guest-disk.img $out/io-guest-disk2.img"
