#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Checks the output of the migration test guests.

usage: expect.py checksum|irq FILE [FILE ...]

The files are joined in order, so a line the source started and the destination finished is
checked whole: pass the source's output and then the destination's.

checksum (checksum-guest.S, debugcon): every line is "STEP SUM OK", SUM is what the checksum
must be after STEP steps, and the steps go up by 0x4000 from line to line.

irq (irq-guest.S, COM1): every line is "REPORT PIT APIC STEP SUM OK", REPORT goes up by one,
PIT is at least 25 * REPORT and never goes back, APIC never goes back, STEP goes up, and SUM is
what the checksum must be after STEP steps. The last line's PIT and APIC counts must be past
the first line's, so both timers kept firing.
"""
import sys

M = 1 << 32
K = 0x01000193


def start_sum(npages):
    s = 0
    for p in range(npages):
        if p % 4:
            b = p * 0x9E3779B1
            # sum_{j<1024} (b + j*K) = 1024*b + K*1023*1024/2
            s += 1024 * b + K * 1023 * 512
    return s % M


def want(s0, i):
    # s0 + sum_{k=1..i} (k*K + 1)
    return (s0 + K * i * (i + 1) // 2 + i) % M


def check_checksum(lines):
    s0 = start_sum(2048)
    bad, prev = 0, None
    for f in lines:
        ok = len(f) == 3 and f[2] == "OK"
        if ok:
            i, s = int(f[0], 16), int(f[1], 16)
            ok = s == want(s0, i) and (prev is None or i == prev + 0x4000)
            prev = i
        if not ok:
            bad += 1
            print("mismatch:", " ".join(f))
    return bad


def check_irq(lines):
    s0 = start_sum(256)
    bad, prev, first = 0, None, None
    for f in lines:
        ok = len(f) == 6 and f[5] == "OK"
        if ok:
            r, pit, apic, step, s = (int(x, 16) for x in f[:5])
            ok = s == want(s0, step) and pit >= 25 * r
            if prev is not None:
                pr, ppit, papic, pstep = prev
                ok = ok and r == pr + 1 and pit >= ppit and apic >= papic and step > pstep
            prev = (r, pit, apic, step)
            if first is None:
                first = prev
        if not ok:
            bad += 1
            print("mismatch:", " ".join(f))
    if first is not None and len(lines) > 1 and (prev[1] <= first[1] or prev[2] <= first[2]):
        bad += 1
        print("timers stopped: first", first, "last", prev)
    return bad


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("checksum", "irq"):
        sys.exit(__doc__)
    text = "".join(open(a, errors="replace").read() for a in sys.argv[2:])
    lines = [line.split() for line in text.splitlines() if line.strip()]
    bad = (check_checksum if sys.argv[1] == "checksum" else check_irq)(lines)
    first = lines[0][0] if lines else "-"
    last = lines[-1][0] if lines else "-"
    print("%d lines, first %s, last %s, %d bad" % (len(lines), first, last, bad))
    sys.exit(1 if bad or not lines else 0)


main()
