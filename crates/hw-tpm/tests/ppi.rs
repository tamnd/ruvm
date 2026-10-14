// SPDX-License-Identifier: GPL-2.0-or-later

//! The memory clear the PPI does on reset, `tpm_ppi_reset()`, over a small address space with
//! RAM, a ROM, nonvolatile memory and the PPI itself.

use std::sync::Arc;

use ruvm_hw_tpm::ppi::TpmPpi;
use ruvm_mem::{AccessCtx, AccessSize, MemTxAttrs, MemorySystem, MmioOps};

const CX: AccessCtx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);

/// The memory overwrite request.
const MOVV: u64 = 0x15a;

#[test]
fn reset_clears_ram_when_asked() {
    let sys = MemorySystem::new();
    let root = sys.new_container("system", 1 << 32).unwrap();
    let space = sys.address_space_init(root, "memory").unwrap();
    let ram = sys.new_ram("ram", 0x10000).unwrap();
    let rom = sys.new_rom("rom", 0x1000).unwrap();
    let nv = sys.new_ram("nvram", 0x1000).unwrap();
    sys.set_nonvolatile(nv, true).unwrap();
    let ppi = Arc::new(TpmPpi::default());
    let ops: Arc<dyn MmioOps> = ppi.clone();
    let dev = sys.new_io("tpm-ppi", 0x400, ops).unwrap();
    sys.add_subregion(root, 0, ram).unwrap();
    sys.add_subregion_overlap(root, 0x8000, dev, 1).unwrap();
    sys.add_subregion(root, 0x20000, rom).unwrap();
    sys.add_subregion(root, 0x30000, nv).unwrap();
    let [ram, rom, nv] = [ram, rom, nv].map(|r| sys.ram_block(r).unwrap());
    for b in [&ram, &rom, &nv] {
        b.fill(0, b.len(), 0xa5).unwrap();
    }

    // Nothing is cleared until the guest asks.
    ppi.reset(&space);
    assert!(!ram.is_zero(0, 0x8000).unwrap());

    ppi.write(&CX, MOVV, AccessSize::B1, 1).unwrap();
    ppi.reset(&space);
    // The RAM the address space maps is cleared, and so is the ROM. The RAM under the PPI
    // and the nonvolatile memory are not.
    assert!(ram.is_zero(0, 0x8000).unwrap());
    let mut hidden = [0; 0x400];
    ram.read(0x8000, &mut hidden).unwrap();
    assert!(hidden.iter().all(|&b| b == 0xa5));
    assert!(ram.is_zero(0x8400, 0x10000 - 0x8400).unwrap());
    assert!(rom.is_zero(0, 0x1000).unwrap());
    assert!(!nv.is_zero(0, 0x1000).unwrap());
    // The request stays for the firmware to see.
    assert!(ppi.memory_clear_requested());
}
