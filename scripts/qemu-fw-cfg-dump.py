#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Dump what a real QEMU puts in fw_cfg, for ruvm's golden tests.

Runs qemu-system-x86_64 with the qtest accelerator, so no guest code runs and no KVM is
needed, and reads every fw_cfg file through the DMA interface the way SeaBIOS does. Before
the ACPI tables are read it programs PMBASE and PCIEXBAR like SeaBIOS would, because the
tables are regenerated on first read from the live device state.

Usage:
    scripts/qemu-fw-cfg-dump.py OUTDIR -- -M q35 -nodefaults [more QEMU options]

OUTDIR gets `dir.txt` (the file directory in slot order: select, size, name), `keys.txt`
(the legacy items this script knows the size of, in hex) and one file per fw_cfg file
under `files/`, keeping the slashes in the names as directories. Option ROMs are left out
and files over 1 KiB lose their trailing zeros (the ACPI blobs are padded to a fixed size).
Set QEMU to pick another binary.
"""

import os
import subprocess
import sys

DESC = 0x1000
BUF = 0x100000

# Legacy items and their sizes: what bios-tables-test and the firmware read.
KEYS = {
    0x00: 4,  # signature
    0x01: 4,  # id
    0x02: 16,  # uuid
    0x03: 8,  # ram size
    0x04: 2,  # nographic
    0x05: 2,  # nb cpus
    0x0e: 2,  # boot menu
    0x0f: 2,  # max cpus
    0x8002: 4,  # irq0 override
    0x8003: 0x1c * 8 + 4,  # hpet, larger than the struct on purpose, padded with zeros
}


class Qtest:
    def __init__(self, argv):
        self.p = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def cmd(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        while True:
            reply = self.p.stdout.readline()
            if not reply:
                raise SystemExit(f"qemu went away after {line!r}")
            if reply.startswith("OK") or reply.startswith("FAIL"):
                break
        if not reply.startswith("OK"):
            raise SystemExit(f"{line!r}: {reply.strip()}")
        return reply.split()[1:]

    def pci_write32(self, devfn, reg, value):
        self.cmd(f"outl 0xcf8 {0x80000000 | (devfn << 8) | reg:#x}")
        self.cmd(f"outl 0xcfc {value:#x}")

    def dma_read(self, key, size):
        """Selects `key` and reads `size` bytes through fw_cfg DMA."""
        if size == 0:
            return b""
        control = (key << 16) | 0x08 | 0x02
        desc = control.to_bytes(4, "big") + size.to_bytes(4, "big") + BUF.to_bytes(8, "big")
        self.cmd(f"write {DESC:#x} 16 0x{desc.hex()}")
        # The DMA register is big endian, the qtest port write little endian.
        self.cmd("outl 0x514 0x0")
        self.cmd(f"outl 0x518 {int.from_bytes(DESC.to_bytes(4, 'big'), 'little'):#x}")
        status = int(self.cmd(f"read {DESC:#x} 4")[0], 16)
        if status != 0:
            raise SystemExit(f"fw_cfg DMA of key {key:#x} failed: {status:#x}")
        return bytes.fromhex(self.cmd(f"read {BUF:#x} {size}")[0][2:])

    def close(self):
        self.p.stdin.close()
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()


def main():
    if "--" not in sys.argv or sys.argv.index("--") != 2:
        raise SystemExit(__doc__)
    out = sys.argv[1]
    qemu = os.environ.get("QEMU", "qemu-system-x86_64")
    argv = [qemu, "-accel", "qtest", "-qtest", "stdio", "-display", "none"] + sys.argv[3:]
    q = Qtest(argv)

    q2 = "q35" in " ".join(sys.argv[3:])
    if q2:
        # What SeaBIOS does first on q35: PMBASE with ACPI enabled, then PCIEXBAR.
        q.pci_write32(0xf8, 0x40, 0x601)
        q.pci_write32(0xf8, 0x44, 0x80)
        q.pci_write32(0x00, 0x64, 0)
        q.pci_write32(0x00, 0x60, 0xB0000001)

    raw = q.dma_read(0x19, 4)
    count = int.from_bytes(raw, "big")
    raw = q.dma_read(0x19, 4 + 64 * count)
    entries = []
    for i in range(count):
        e = raw[4 + 64 * i : 4 + 64 * (i + 1)]
        size = int.from_bytes(e[0:4], "big")
        select = int.from_bytes(e[4:6], "big")
        name = e[8:].split(b"\0")[0].decode()
        entries.append((select, size, name))

    os.makedirs(os.path.join(out, "files"), exist_ok=True)
    with open(os.path.join(out, "dir.txt"), "w") as f:
        for select, size, name in entries:
            f.write(f"{select:#06x} {size} {name}\n")
    for select, size, name in entries:
        # Option ROMs are firmware files, not something the machine builds.
        if name.startswith("genroms/") or name.startswith("vgaroms/"):
            continue
        data = q.dma_read(select, size)
        # The ACPI blobs are padded to a fixed size; dir.txt keeps the real size.
        if size > 1024:
            data = data.rstrip(b"\0")
        path = os.path.join(out, "files", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    with open(os.path.join(out, "keys.txt"), "w") as f:
        for key, size in KEYS.items():
            f.write(f"{key:#06x} {q.dma_read(key, size).hex()}\n")
    q.close()


if __name__ == "__main__":
    main()
