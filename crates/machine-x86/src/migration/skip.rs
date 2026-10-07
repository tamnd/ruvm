// SPDX-License-Identifier: GPL-2.0-or-later

//! The q35 sections ruvm accepts from QEMU without loading them: the stream layout, so the
//! stream is parsed and checked, with the state dropped. They are never sent, and a QEMU
//! destination keeps its reset state for them.

use std::sync::LazyLock;

use ruvm_migration::SaveVm;
use ruvm_vmstate::{VmStateDescription, VmStateField};

/// The state of a skipped section: nothing.
#[derive(Debug, Default)]
pub(crate) struct Skip;

/// `vmstate_kvmvapic`: the TPR patching state of kvmvapic, 144 bytes. It only does anything
/// under KVM.
static KVM_TPR_OPT: LazyLock<VmStateDescription<Skip>> = LazyLock::new(|| {
    VmStateDescription::new("kvm-tpr-opt")
        .version_id(1)
        .minimum_version_id(1)
        .field(VmStateField::unused(144))
});

/// Registers the `kvm-tpr-opt` section of kvmvapic, which comes with the first APIC.
pub(crate) fn register_vapic(savevm: &mut SaveVm) {
    savevm.register_discard_with("", Some(0), &KVM_TPR_OPT, Skip::default);
}
