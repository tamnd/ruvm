// SPDX-License-Identifier: GPL-2.0-or-later

//! The `fw_cfg` section, from hw/nvram/fw_cfg.c.

use std::sync::{Arc, LazyLock};

use ruvm_base::Result;
use ruvm_hw_core::fw_cfg::{FwCfgState, FwCfgVmState};
use ruvm_migration::SaveVm;
use ruvm_vmstate::{StreamReader, StreamWriter, VmStateDescription, VmStateField, VmStateInfo};

/// `vmstate_hack_uint32_as_uint16`: the 16 bit `cur_offset` of version 1 streams, loaded into
/// the 32 bit field. Its test only passes for version 1, so it is never saved.
#[derive(Debug, Clone, Copy, Default)]
struct Uint32AsUint16;

impl VmStateInfo<u32> for Uint32AsUint16 {
    fn name(&self) -> &'static str {
        "int32_as_uint16"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut u32, _size: usize) -> Result<()> {
        *v = u32::from(f.get_be16());
        Ok(())
    }

    fn save(&self, _f: &mut StreamWriter, _v: &u32, _size: usize) -> Result<()> {
        // put_unused() writes nothing.
        Ok(())
    }
}

type S = FwCfgVmState;

/// `vmstate_fw_cfg_dma`.
static VMSTATE_FW_CFG_DMA: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("fw_cfg/dma")
        .needed(S::dma_needed)
        .field(VmStateField::scalar("dma_addr", |s: &mut S| &mut s.dma_addr))
});

/// `vmstate_fw_cfg_acpi_mr`. The post_load, which resizes the ACPI ROM regions, is
/// [`FwCfgState::vmstate_load`] keeping the sizes.
static VMSTATE_FW_CFG_ACPI_MR: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("fw_cfg/acpi_mr")
        .version_id(1)
        .minimum_version_id(1)
        .needed(S::acpi_mr_needed)
        .fields([
            VmStateField::scalar("table_mr_size", |s: &mut S| &mut s.table_mr_size),
            VmStateField::scalar("linker_mr_size", |s: &mut S| &mut s.linker_mr_size),
            VmStateField::scalar("rsdp_mr_size", |s: &mut S| &mut s.rsdp_mr_size),
        ])
});

/// `vmstate_fw_cfg`.
pub(crate) static VMSTATE_FW_CFG: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("fw_cfg")
        .version_id(2)
        .minimum_version_id(1)
        .fields([
            VmStateField::scalar("cur_entry", |s: &mut S| &mut s.cur_entry),
            VmStateField::single("cur_offset", &Uint32AsUint16, |s: &mut S| &mut s.cur_offset)
                .test(|_, version_id| version_id == 1),
            VmStateField::scalar("cur_offset", |s: &mut S| &mut s.cur_offset).version(2),
        ])
        .subsection(&VMSTATE_FW_CFG_DMA)
        .subsection(&VMSTATE_FW_CFG_ACPI_MR)
});

/// Registers the `fw_cfg` section of `fw_cfg`, instance 0 with no path prefix as in QEMU.
pub(crate) fn register(savevm: &mut SaveVm, fw_cfg: &Arc<FwCfgState>) {
    let (get, put) = (Arc::clone(fw_cfg), Arc::clone(fw_cfg));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_FW_CFG,
        move || Ok(get.vmstate_save()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
}

#[cfg(test)]
mod tests {
    use ruvm_vmstate::{vmstate_load_state, vmstate_save_state};

    use super::*;

    #[test]
    fn fw_cfg_layout_matches_qemu() {
        let mut s = S { cur_entry: 0x19, cur_offset: 0x1234, dma_addr: 0x10, ..S::default() };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_FW_CFG, &mut s).unwrap();
        assert_eq!(f.as_bytes(), &[0, 0x19, 0, 0, 0x12, 0x34]);

        // With DMA the address follows in a subsection.
        s.dma_enabled = true;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_FW_CFG, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 6 + 2 + "fw_cfg/dma".len() + 4 + 8);
        assert_eq!(&b[b.len() - 8..], &0x10u64.to_be_bytes());
        let mut back = S { dma_enabled: true, ..S::default() };
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_FW_CFG, &mut back, 2).unwrap();
        assert_eq!(back, s);

        // An unaligned RSDP size brings the ACPI MR sizes along.
        s.rsdp_mr_size = 36;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_FW_CFG, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(&b[b.len() - 8..], &36u64.to_be_bytes());
        let mut back = S { dma_enabled: true, ..S::default() };
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_FW_CFG, &mut back, 2).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn fw_cfg_version_1_has_a_16_bit_offset() {
        let b = [0, 0x19, 0xab, 0xcd];
        let mut back = S::default();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_FW_CFG, &mut back, 1).unwrap();
        assert_eq!((back.cur_entry, back.cur_offset), (0x19, 0xabcd));
    }
}
