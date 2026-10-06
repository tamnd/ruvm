#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Remake the risu images and traces that ruvm's risu harnesses replay.
#
#   regen.sh small               remake the checked in ones, in
#                                crates/target-arm/tests/data/risu and
#                                crates/machine-x86/tests/data/risu
#   regen.sh big DIR [SCALE]     make ones with SCALE (default 20) times as many instructions
#                                in DIR on RISU_HOST, too big for git; run the harnesses there
#                                with RUVM_RISU_DIR=DIR
#
# RISU_HOST is an ssh host running x86-64 Linux, with the tools risu.sh needs; it builds
# risu, makes the images and records the x86-64 traces, so it must be the hardware the x86
# traces are meant to describe. ARM64_SHELL is a command that runs its arguments as a
# command on an AArch64 Linux machine with stdin passed through, by default `colima ssh --`
# (colima's VM runs natively on Apple silicon); the AArch64 traces record that machine.
# Images and traces stream between the two through this machine without being stored on it.
# RISU_WORK is where risu is built on RISU_HOST, relative to the home directory.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
host=${RISU_HOST:?set RISU_HOST to an x86-64 Linux ssh host}
work=${RISU_WORK:-ruvm-risu}
read -r -a arm <<< "${ARM64_SHELL:-colima ssh --}"
A64="a64 a64_v a64_ext"

case ${1:-} in
small) out=$work/small scale=1 ;;
big) out=${2:?regen.sh big DIR [SCALE]} scale=${3:-20} ;;
*)
    sed -n '4,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac

ssh "$host" "mkdir -p $work/scripts"
tar --no-xattrs -C "$HERE" -cf - . | ssh "$host" "tar -C $work/scripts -xf -"
ssh "$host" "bash $work/scripts/risu.sh setup $work > $work/setup.log 2>&1" ||
    { echo "risu.sh setup failed, see $work/setup.log on $host" >&2; exit 1; }
ssh "$host" "rm -rf $out && bash $work/scripts/risu.sh gen $work $out $scale > /dev/null 2>&1 &&
    bash $work/scripts/risu.sh record-x86 $work $out > /dev/null"

# The AArch64 traces, recorded in a scratch directory on the AArch64 machine.
bins=$(for g in $A64; do printf '%s.bin ' "$g"; done)
ssh "$host" "tar -C $out -cf - $bins -C ~/$work/build/arm64 risu" |
    "${arm[@]}" sh -c "set -e; d=/tmp/ruvm-risu; rm -rf \$d; mkdir -p \$d; cd \$d; tar xf -
        for g in $A64; do ./risu --master -t \$g.trace \$g.bin > \$g.log 2>&1; done"
"${arm[@]}" sh -c 'cd /tmp/ruvm-risu && tar cf - *.trace && rm -rf /tmp/ruvm-risu' |
    ssh "$host" "tar -C $out -xf -"

if [ "$1" = big ]; then
    ssh "$host" "ls -l $out"
    exit 0
fi

# Install the small ones, gzipped: risu writes its traces gzipped already.
install() {
    local dest=$1 pat=$2 f
    rm -f "$dest"/*.gz
    mkdir -p "$dest"
    ssh "$host" "cd $out && tar cf - $pat" | tar -C "$dest" -xf -
    for f in "$dest"/*.bin; do gzip -9nf "$f"; done
    for f in "$dest"/*.trace; do
        if [ "$(head -c 2 "$f" | od -An -tx1 | tr -d ' ')" = 1f8b ]; then
            mv "$f" "$f.gz"
        else
            gzip -9nf "$f"
        fi
    done
    ls -l "$dest"
}
install "$ROOT/crates/target-arm/tests/data/risu" "a64*.bin a64*.trace"
install "$ROOT/crates/machine-x86/tests/data/risu" "x86_*.bin x86_*.trace"
