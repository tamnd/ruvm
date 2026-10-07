#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""usage: io-expect.py FILE [FILE ...]

Checks the serial output of io-guest.c, the files joined in order as one run across
migrations. Lines not starting with "IO " are kernel messages and are skipped. The run must
have the START line, each step one more than the last, the block, tx and rx counts never
going down, rx going up over the run and within the last file, and no failures.
"""

import sys


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    lines, bad, starts = [], 0, 0
    rx_last = []
    for i, a in enumerate(sys.argv[1:]):
        for line in open(a, errors="replace").read().splitlines():
            line = line.strip("\r\x00")
            if not line.startswith("IO "):
                continue
            f = line.split()
            if f[1] == "START":
                starts += 1
                if starts > 1:
                    print("mismatch: second START:", line)
                    bad += 1
                continue
            if len(f) < 7 or f[6] != "OK":
                print("mismatch: failure:", line)
                bad += 1
                continue
            try:
                vals = [int(x, 16) for x in f[1:6]]
            except ValueError:
                print("mismatch: garbled:", line)
                bad += 1
                continue
            if vals[4]:
                print("mismatch: bad count:", line)
                bad += 1
            if lines:
                prev = lines[-1]
                if vals[0] != prev[0] + 1:
                    print("mismatch: step %x after %x" % (vals[0], prev[0]))
                    bad += 1
                for k, name in ((1, "blocks"), (2, "tx"), (3, "rx")):
                    if vals[k] < prev[k]:
                        print("mismatch: %s %x after %x" % (name, vals[k], prev[k]))
                        bad += 1
            lines.append(vals)
            if i == len(sys.argv) - 2:
                rx_last.append(vals[3])
    if not starts:
        print("mismatch: no START line")
        bad += 1
    if lines and lines[-1][3] <= lines[0][3]:
        print("mismatch: rx did not go up over the run")
        bad += 1
    if len(rx_last) < 2 or rx_last[-1] <= rx_last[0]:
        print("mismatch: rx did not go up in the last file")
        bad += 1
    first = "%x" % lines[0][0] if lines else "-"
    last = "%x" % lines[-1][0] if lines else "-"
    print("%d lines, first %s, last %s, blocks %d, tx %d, rx %d, %d bad"
          % (len(lines), first, last, lines[-1][1] if lines else 0,
             lines[-1][2] if lines else 0, lines[-1][3] if lines else 0, bad))
    sys.exit(1 if bad or not lines else 0)


main()
