#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Fetch and build Berkeley SoftFloat 3e and TestFloat 3e, then print the path of
# testfloat_gen. Use it as:
#
#   export RUVM_TESTFLOAT_GEN="$(crates/softfloat/tests/testfloat/build.sh)"
#   cargo test -p ruvm-softfloat --test testfloat -- --ignored --nocapture
#
# The first argument is the work directory (default: $TMPDIR/ruvm-testfloat).
# SoftFloat is built with its 8086-SSE specialization, which is what the
# Linux-x86_64-GCC build directory selects. The makefiles call `gcc`, which is
# clang on macOS.
set -eu

dir="${1:-${TMPDIR:-/tmp}/ruvm-testfloat}"
url="http://www.jhauser.us/arithmetic"
mkdir -p "$dir"
cd "$dir"

for pkg in SoftFloat-3e TestFloat-3e; do
    if [ ! -d "$pkg" ]; then
        curl -fsSL -o "$pkg.zip" "$url/$pkg.zip" >&2
        unzip -q "$pkg.zip" >&2
    fi
done

make -s -C SoftFloat-3e/build/Linux-x86_64-GCC softfloat.a >&2
make -s -C TestFloat-3e/build/Linux-x86_64-GCC testfloat_gen >&2
echo "$dir/TestFloat-3e/build/Linux-x86_64-GCC/testfloat_gen"
