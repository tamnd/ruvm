// SPDX-License-Identifier: GPL-2.0-or-later

//! The `fw_cfg` section, from hw/nvram/fw_cfg.c. Its description is
//! [`ruvm_hw_core::vmstate::VMSTATE_FW_CFG`].

use std::sync::Arc;

use ruvm_hw_core::fw_cfg::FwCfgState;
use ruvm_hw_core::vmstate::VMSTATE_FW_CFG;
use ruvm_migration::SaveVm;

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
