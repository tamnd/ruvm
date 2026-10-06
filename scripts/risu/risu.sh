#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Build risu and make the risu images and x86-64 traces that ruvm's risu harnesses replay
# (crates/target-arm/tests/risu.rs and crates/machine-x86/tests/risu.rs). Runs on an x86-64
# Linux host with git, curl, perl, make, gcc and aarch64-linux-gnu-gcc; the x86-64 traces
# record the host CPU, so use the machine the traces are meant to describe (an AMD EPYC with
# AVX2 and BMI2 for the checked in ones). regen.sh drives this and records the AArch64
# traces on an AArch64 Linux machine.
#
#   risu.sh setup WORK             fetch and build risu into WORK
#   risu.sh gen WORK OUT SCALE     make the images of every group in OUT, with SCALE times
#                                  the instructions of the checked in ones
#   risu.sh record-x86 WORK OUT    record OUT/x86_*.trace from OUT/x86_*.bin
#
# risu is pm215's tree at RISU_REV with Jan Bobek's x86 series (v3, July 2019, never
# merged) and the patches in patches/ on top.

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
RISU_URL=https://gitlab.com/pm215/risu.git
RISU_REV=eed2249
X86_SERIES=https://patchew.org/QEMU/20190711223300.6061-1-jan.bobek@gmail.com/mbox
LIST_COMPARE=https://cpan.metacpan.org/authors/id/J/JK/JKEENAN/List-Compare-0.55.tar.gz
ZLIB=https://zlib.net/fossils/zlib-1.3.1.tar.gz

# The groups, with the number of instructions in the checked in image. The AArch64 groups
# leave out what ruvm's `max` CPU does not have (its ID registers do not advertise them, so
# ruvm UNDEFs them): LRCPC2 (STLUR, LDAPUR), SHA512 and SHA3, and the v8.3 complex number
# instructions (FCADD, FCMLA). The Apple M4 has no SVE, so there is no SVE group.
RISU_GROUPS="a64 a64_v a64_ext x86_int x86_bmi x86_sse x86_avx"

# Set N and ARGS to the instruction count and the risugen arguments of the group $1.
group() {
    local any='[A-Za-z0-9_]+ ' x86=(--x86_64 --xfeatures sse)
    case $1 in
    a64) N=1000 ARGS=(--pattern "${any}A64" --not-pattern STLUR,LDAPUR,LDAPURS64,LDAPURS32
        aarch64.risu) ;;
    a64_v) N=1000 ARGS=(--pattern "${any}A64_V"
        --not-pattern SHA512H,SHA512H2,SHA512SU0,SHA512SU1,RAX1,EOR3,BCAX,XAR aarch64.risu) ;;
    a64_ext) N=1000 ARGS=(--pattern "${any}A64_V8[0-2]" aarch64.risu) ;;
    x86_int) N=1000 ARGS=("${x86[@]}" --pattern "${any}INT" "$HERE/x86_int.risu") ;;
    x86_bmi) N=300 ARGS=("${x86[@]}" --pattern "${any}(BMI1|BMI2|ABM|POPCNT|ADX|MOVBE|CRC32)"
        "$HERE/x86_int.risu") ;;
    x86_sse) N=500 ARGS=("${x86[@]}"
        --pattern "${any}(MMX|SSE|SSE2|SSE3|SSSE3|SSE4_1|SSE4_2|AES|PCLMULQDQ)" x86.risu) ;;
    x86_avx) N=500 ARGS=(--x86_64 --xfeatures avx
        --pattern "${any}(AVX|AVX2|AES_AVX|PCLMULQDQ_AVX)" x86.risu) ;;
    esac
}

setup() {
    mkdir -p "$1"
    local work
    work=$(cd "$1" && pwd)
    cd "$work"
    # The downloads are kept; the risu tree is made again so that it has the patches as
    # they are now.
    [ -d risu.git ] || git clone -q --bare "$RISU_URL" risu.git
    [ -f x86v3.mbox ] || curl -sfL "$X86_SERIES" -o x86v3.mbox
    rm -rf risu
    git clone -q risu.git risu
    git -C risu checkout -q -b ruvm "$RISU_REV"
    git -C risu -c user.name=ruvm -c user.email=ruvm@localhost am -q "$work/x86v3.mbox"
    git -C risu -c user.name=ruvm -c user.email=ruvm@localhost am -q "$HERE"/patches/*.patch
    if [ ! -d List-Compare-0.55 ]; then
        curl -sfL "$LIST_COMPARE" | tar xzf -
    fi
    if [ ! -d zlib-arm64 ]; then
        curl -sfL "$ZLIB" | tar xzf -
        (cd zlib-1.3.1 && CC=aarch64-linux-gnu-gcc ./configure --static --prefix="$work/zlib-arm64" >/dev/null &&
            make -s -j8 install >/dev/null)
    fi
    local s=$work/risu
    rm -rf build
    mkdir -p build/x86_64 build/arm64
    (cd build/x86_64 && "$s"/configure --static >/dev/null && make -s >/dev/null)
    printf '#define HAVE_ZLIB 1\n' > build/arm64/config.h
    aarch64-linux-gnu-gcc -static -O1 -Wall -D_GNU_SOURCE -DARCH=aarch64 -Uaarch64 \
        -Ibuild/arm64 -I"$s" -I"$work/zlib-arm64/include" -o build/arm64/risu \
        "$s"/risu.c "$s"/comms.c "$s"/risu_aarch64.c "$s"/risu_reginfo_aarch64.c \
        -L"$work/zlib-arm64/lib" -lz
    echo "built $work/build/x86_64/risu and $work/build/arm64/risu"
}

gen() {
    local work out scale=$3
    work=$(cd "$1" && pwd)
    mkdir -p "$2"
    out=$(cd "$2" && pwd)
    cd "$work/risu"
    export PERL5LIB=$work/List-Compare-0.55/lib
    local name
    for name in $RISU_GROUPS; do
        group "$name"
        ./risugen --numinsns $((N * scale)) "${ARGS[@]}" "$out/$name.bin" > /dev/null
        echo "$out/$name.bin"
    done
}

record_x86() {
    local work=$1 out=$2 f name xf
    for f in "$out"/x86_*.bin; do
        name=${f%.bin}
        xf=sse
        case $name in *_avx) xf=avx ;; esac
        "$work/build/x86_64/risu" --xfeatures $xf --master -t "$name.trace" "$f" > "$name.log" 2>&1
        echo "$name.trace"
    done
}

case ${1:-} in
setup) setup "$2" ;;
gen) gen "$2" "$3" "$4" ;;
record-x86) record_x86 "$2" "$3" ;;
*)
    sed -n '4,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
