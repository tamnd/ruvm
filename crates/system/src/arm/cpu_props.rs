// SPDX-License-Identifier: GPL-2.0-or-later

//! The Arm CPU properties `query-cpu-model-expansion` and `-cpu` set: `aarch64`, `pmu`, `sve`,
//! the `sve<N>` vector lengths and the pointer authentication ones, from target/arm's
//! arm-qmp-cmds.c, cpu.c and cpu64.c.
//!
//! Every CPU type of qemu-system-aarch64 is known here with the properties its object has,
//! though ruvm runs only some of them. The properties are set in the order QEMU sets them, and
//! then `arm_cpu_finalize_features()` works out the vector lengths and checks the pointer
//! authentication choice, with the same errors. Without TCG (under `-accel qtest`) `max` is
//! the `host` like model QEMU makes then, with no SVE or pointer authentication properties.

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{CpuModelExpansionInfo, CpuModelExpansionType, CpuModelInfo};
use ruvm_qapi::{QDict, QValue};
use ruvm_target_arm::cpu::{ARM_MAX_VQ, PauthAlg};

/// The power-of-two vector lengths in a vector length map, one bit per quadword count less
/// one (`SVE_VQ_POW2_MAP`).
const SVE_VQ_POW2_MAP: u32 = (1 << 15) | (1 << 7) | (1 << 3) | (1 << 1) | 1;

/// `cpu_model_advertised_features[]`: the properties the expansion sets and reports, in the
/// order it sets them.
const ADVERTISED: &[&str] = &[
    "aarch64",
    "pmu",
    "sve",
    "sve128",
    "sve256",
    "sve384",
    "sve512",
    "sve640",
    "sve768",
    "sve896",
    "sve1024",
    "sve1152",
    "sve1280",
    "sve1408",
    "sve1536",
    "sve1664",
    "sve1792",
    "sve1920",
    "sve2048",
    "kvm-no-adjvtime",
    "kvm-steal-time",
    "pauth",
    "pauth-impdef",
    "pauth-qarma3",
    "pauth-qarma5",
];

/// Whether a CPU type has the `aarch64` property and whether TCG can turn it off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Aarch64 {
    /// An AArch32 only CPU, without the property.
    No,
    /// AArch64 only at the highest exception level, so the property cannot be turned off.
    Fixed,
    /// AArch32 at the highest exception level too, so TCG can turn AArch64 off.
    Optional,
}

/// What the object of one CPU type has.
struct CpuType {
    /// The type name without the `-arm-cpu` suffix.
    name: &'static str,
    aarch64: Aarch64,
    /// The `pmu` property, which the CPUs with a PMU have.
    pmu: bool,
    /// The vector lengths the CPU supports, zero without SVE.
    sve: u32,
    /// The pointer authentication properties.
    pauth: bool,
}

const fn ty(name: &'static str, aarch64: Aarch64, pmu: bool, sve: u32, pauth: bool) -> CpuType {
    CpuType { name, aarch64, pmu, sve, pauth }
}

/// The CPU types of qemu-system-aarch64, as `query-cpu-definitions` lists them.
const TYPES: &[CpuType] = &[
    ty("a64fx", Aarch64::Fixed, true, 0xb, false),
    ty("arm1026", Aarch64::No, false, 0, false),
    ty("arm1136", Aarch64::No, false, 0, false),
    ty("arm1136-r2", Aarch64::No, false, 0, false),
    ty("arm1176", Aarch64::No, false, 0, false),
    ty("arm11mpcore", Aarch64::No, false, 0, false),
    ty("arm926", Aarch64::No, false, 0, false),
    ty("arm946", Aarch64::No, false, 0, false),
    ty("cortex-a15", Aarch64::No, true, 0, false),
    ty("cortex-a35", Aarch64::Optional, true, 0, false),
    ty("cortex-a53", Aarch64::Optional, true, 0, false),
    ty("cortex-a55", Aarch64::Optional, true, 0, false),
    ty("cortex-a57", Aarch64::Optional, true, 0, false),
    ty("cortex-a7", Aarch64::No, true, 0, false),
    ty("cortex-a710", Aarch64::Fixed, true, 0x1, true),
    ty("cortex-a72", Aarch64::Optional, true, 0, false),
    ty("cortex-a76", Aarch64::Fixed, true, 0, false),
    ty("cortex-a78ae", Aarch64::Fixed, true, 0, false),
    ty("cortex-a8", Aarch64::No, true, 0, false),
    ty("cortex-a9", Aarch64::No, true, 0, false),
    ty("cortex-m0", Aarch64::No, false, 0, false),
    ty("cortex-m3", Aarch64::No, false, 0, false),
    ty("cortex-m33", Aarch64::No, false, 0, false),
    ty("cortex-m4", Aarch64::No, false, 0, false),
    ty("cortex-m55", Aarch64::No, false, 0, false),
    ty("cortex-m7", Aarch64::No, false, 0, false),
    ty("cortex-r5", Aarch64::No, true, 0, false),
    ty("cortex-r52", Aarch64::No, false, 0, false),
    ty("cortex-r5f", Aarch64::No, true, 0, false),
    ty("max", Aarch64::Optional, true, 0xffff, true),
    ty("neoverse-n1", Aarch64::Fixed, true, 0, false),
    ty("neoverse-n2", Aarch64::Fixed, true, 0x1, true),
    ty("neoverse-v1", Aarch64::Fixed, true, 0x3, true),
    ty("sa1100", Aarch64::No, false, 0, false),
    ty("sa1110", Aarch64::No, false, 0, false),
    ty("ti925t", Aarch64::No, false, 0, false),
];

/// `MAKE_64BIT_MASK(0, n)` for a vector length map.
fn low_bits(n: u32) -> u32 {
    if n >= 32 { u32::MAX } else { (1 << n) - 1 }
}

/// The quadword count of an `sve<N>` property name.
fn sve_vq(prop: &str) -> Option<u32> {
    let bits: u32 = prop.strip_prefix("sve")?.parse().ok()?;
    let vq = bits / 128;
    (bits % 128 == 0 && (1..=ARM_MAX_VQ as u32).contains(&vq) && prop == format!("sve{bits}"))
        .then_some(vq)
}

/// The properties of one Arm CPU object.
#[derive(Debug)]
pub(crate) struct ArmCpuProps {
    name: &'static str,
    /// Under TCG, which decides what `max` is and whether `aarch64` can be turned off.
    tcg: bool,
    has_aarch64: Aarch64,
    has_pmu: bool,
    /// The vector lengths the CPU supports (`sve_vq.supported`), zero without SVE.
    sve_supported: u32,
    has_pauth: bool,
    aarch64: bool,
    pmu: bool,
    /// ID_AA64PFR0_EL1.SVE, which the `sve` property sets.
    sve: bool,
    /// The `sve<N>` properties set to on (`sve_vq.map`), and after finalizing the vector
    /// lengths the CPU has.
    sve_map: u32,
    /// The `sve<N>` properties set at all (`sve_vq.init`).
    sve_init: u32,
    /// `sve-max-vq`, zero if not set, and after finalizing the largest vector length.
    sve_max_vq: u32,
    pauth: bool,
    pauth_impdef: bool,
    pauth_qarma3: bool,
    pauth_qarma5: bool,
}

impl ArmCpuProps {
    /// A new object of the CPU type `name`, or `None` if there is no such type.
    pub(crate) fn new(name: &str, tcg: bool) -> Option<ArmCpuProps> {
        let t = TYPES.iter().find(|t| t.name == name)?;
        // aarch64_max_initfn() makes `max` like `host` without TCG.
        let tcg_max = tcg || t.name != "max";
        let (sve_supported, has_pauth) = if tcg_max { (t.sve, t.pauth) } else { (0, false) };
        Some(ArmCpuProps {
            name: t.name,
            tcg,
            has_aarch64: t.aarch64,
            has_pmu: t.pmu,
            sve_supported,
            has_pauth,
            aarch64: t.aarch64 != Aarch64::No,
            pmu: t.pmu,
            sve: sve_supported != 0,
            sve_map: 0,
            sve_init: 0,
            sve_max_vq: 0,
            pauth: has_pauth,
            pauth_impdef: false,
            pauth_qarma3: false,
            pauth_qarma5: false,
        })
    }

    /// Whether the object has the boolean property `prop`.
    pub(crate) fn has(&self, prop: &str) -> bool {
        match prop {
            "aarch64" => self.has_aarch64 != Aarch64::No,
            "pmu" => self.has_pmu,
            "sve" => self.sve_supported != 0,
            "pauth" | "pauth-impdef" | "pauth-qarma3" | "pauth-qarma5" => self.has_pauth,
            _ => self.sve_supported != 0 && sve_vq(prop).is_some(),
        }
    }

    /// The error for a property the object does not have.
    fn check(&self, prop: &str) -> Result<()> {
        if self.has(prop) {
            return Ok(());
        }
        Err(Error::generic(format!("Property '{}-arm-cpu.{prop}' not found", self.name)))
    }

    /// `object_property_set_bool()`.
    pub(crate) fn set(&mut self, prop: &str, value: bool) -> Result<()> {
        self.check(prop)?;
        match prop {
            // aarch64_cpu_set_aarch64()
            "aarch64" => {
                if !value && !self.tcg {
                    return Err(Error::generic(
                        "'aarch64' feature cannot be disabled for this accelerator",
                    ));
                }
                if !value && self.has_aarch64 == Aarch64::Fixed {
                    return Err(Error::generic(
                        "'aarch64' feature cannot be disabled for this TCG CPU because it does \
                         not support 32-bit execution at its highest implemented exception level",
                    ));
                }
                self.aarch64 = value;
            }
            "pmu" => self.pmu = value,
            "sve" => self.sve = value,
            "pauth" => self.pauth = value,
            "pauth-impdef" => self.pauth_impdef = value,
            "pauth-qarma3" => self.pauth_qarma3 = value,
            "pauth-qarma5" => self.pauth_qarma5 = value,
            // cpu_arm_set_vq()
            _ => {
                let bit = 1 << (sve_vq(prop).expect("checked by has()") - 1);
                if value {
                    self.sve_map |= bit;
                } else {
                    self.sve_map &= !bit;
                }
                self.sve_init |= bit;
            }
        }
        Ok(())
    }

    /// `sve-max-vq`, already checked to be from 1 to `ARM_MAX_VQ`.
    pub(crate) fn set_sve_max_vq(&mut self, vq: u32) {
        self.sve_max_vq = vq;
    }

    /// `arm_cpu_finalize_features()`: the SVE vector lengths, then the pointer authentication
    /// choice, both only for a CPU in AArch64.
    pub(crate) fn finalize(&mut self) -> Result<()> {
        if !self.aarch64 {
            return Ok(());
        }
        self.sve_finalize()?;
        self.pauth_finalize()
    }

    /// `aarch64_cpu_sve_finalize()` for TCG.
    fn sve_finalize(&mut self) -> Result<()> {
        let (init, supported) = (self.sve_init, self.sve_supported);
        let mut map = self.sve_map;
        let (mut max_vq, mut mask) = (0, 0);
        if map != 0 {
            // Explicit sve<N> properties: the largest one is the maximum.
            max_vq = 32 - map.leading_zeros();
            mask = low_bits(max_vq);
            if self.sve_max_vq != 0 && max_vq > self.sve_max_vq {
                return Err(Error::generic(format!("cannot enable sve{}", max_vq * 128)).hint(
                    format!(
                        "sve{} is larger than the maximum vector length, sve-max-vq={} ({} bits)\n",
                        max_vq * 128,
                        self.sve_max_vq,
                        self.sve_max_vq * 128
                    ),
                ));
            }
            // Propagate enabled bits down through required powers-of-two.
            map |= SVE_VQ_POW2_MAP & !init & mask;
        } else if self.sve_max_vq == 0 {
            if !self.sve {
                return Ok(());
            }
            // Disabling a power-of-two disables all larger lengths.
            let vq = (init & SVE_VQ_POW2_MAP).trailing_zeros() + 1;
            max_vq = if vq <= ARM_MAX_VQ as u32 { vq - 1 } else { ARM_MAX_VQ as u32 };
            mask = low_bits(max_vq);
            map = supported & !init & mask;
            if map == 0 {
                return Err(Error::generic(format!("cannot disable sve{}", vq * 128))
                    .hint(format!(
                        "Disabling sve{} results in all vector lengths being disabled.\n",
                        vq * 128
                    ))
                    .hint("With SVE enabled, at least one vector length must be enabled.\n"));
            }
            max_vq = 32 - map.leading_zeros();
            mask = low_bits(max_vq);
        }
        if self.sve_max_vq != 0 {
            max_vq = self.sve_max_vq;
            mask = low_bits(max_vq);
            if init & !map & (1 << (max_vq - 1)) != 0 {
                return Err(Error::generic(format!("cannot disable sve{}", max_vq * 128)).hint(
                    format!(
                        "The maximum vector length must be enabled, sve-max-vq={max_vq} ({} bits)\n",
                        max_vq * 128
                    ),
                ));
            }
            // Set all bits not explicitly set within sve-max-vq.
            map |= !init & mask;
        }
        map &= mask;

        // Ensure the set of lengths matches what is supported.
        let diff = map ^ (supported & mask);
        if diff != 0 {
            let vq = 32 - diff.leading_zeros();
            if map & (1 << (vq - 1)) != 0 {
                let unsupported =
                    format!("This CPU does not support the vector length {}-bits.\n", vq * 128);
                if self.sve_max_vq != 0 {
                    return Err(Error::generic(format!(
                        "cannot set sve-max-vq={}",
                        self.sve_max_vq
                    ))
                    .hint(unsupported)
                    .hint(
                        "It may not be possible to use sve-max-vq with this CPU. Try using only \
                         sve<N> properties.\n",
                    ));
                }
                return Err(
                    Error::generic(format!("cannot enable sve{}", vq * 128)).hint(unsupported)
                );
            }
            // Ensure all required powers-of-two are enabled.
            let missing = SVE_VQ_POW2_MAP & mask & !map;
            if missing != 0 {
                let vq = 32 - missing.leading_zeros();
                return Err(Error::generic(format!("cannot disable sve{}", vq * 128)).hint(
                    format!(
                        "sve{} is required as it is a power-of-two length smaller than the \
                         maximum, sve{}\n",
                        vq * 128,
                        max_vq * 128
                    ),
                ));
            }
        }

        if !self.sve {
            return Err(Error::generic(format!("cannot enable sve{}", max_vq * 128))
                .hint("SVE must be enabled to enable vector lengths.\n")
                .hint("Add sve=on to the CPU property list.\n"));
        }
        self.sve_max_vq = max_vq;
        self.sve_map = map;
        Ok(())
    }

    /// `aarch64_cpu_pauth_finalize()` for TCG.
    fn pauth_finalize(&self) -> Result<()> {
        if !self.has_pauth {
            return Ok(());
        }
        let algs = [self.pauth_impdef, self.pauth_qarma3, self.pauth_qarma5];
        if self.pauth {
            if algs.iter().filter(|&&a| a).count() > 1 {
                return Err(Error::generic(
                    "cannot enable pauth-impdef, pauth-qarma3 and pauth-qarma5 at the same time",
                ));
            }
        } else if algs.contains(&true) {
            return Err(Error::generic(
                "cannot enable pauth-impdef, pauth-qarma3 or pauth-qarma5 without pauth",
            )
            .hint("Add pauth=on to the CPU property list.\n"));
        }
        Ok(())
    }

    /// The `aarch64` property.
    pub(crate) fn aarch64(&self) -> bool {
        self.aarch64
    }

    /// The `pmu` property.
    pub(crate) fn pmu(&self) -> bool {
        self.pmu
    }

    /// After [`Self::finalize`], the vector lengths (`sve_vq.map`), `None` without SVE.
    pub(crate) fn sve_vq_map(&self) -> Option<u32> {
        (self.sve_supported != 0 && self.sve).then_some(self.sve_map)
    }

    /// After [`Self::finalize`], the pointer authentication algorithm, `None` with `pauth`
    /// off. With none chosen it is the IMPDEF one, as for the machine types after 9.2.
    pub(crate) fn pauth_alg(&self) -> Option<PauthAlg> {
        if !self.pauth {
            None
        } else if self.pauth_qarma5 {
            Some(PauthAlg::Qarma5)
        } else if self.pauth_qarma3 {
            Some(PauthAlg::Qarma3)
        } else {
            Some(PauthAlg::Impdef)
        }
    }

    /// The value of the property `prop`, which the object has.
    fn get(&self, prop: &str) -> bool {
        match prop {
            "aarch64" => self.aarch64,
            "pmu" => self.pmu,
            "sve" => self.sve,
            "pauth" => self.pauth,
            "pauth-impdef" => self.pauth_impdef,
            "pauth-qarma3" => self.pauth_qarma3,
            "pauth-qarma5" => self.pauth_qarma5,
            // cpu_arm_get_vq(): all vector lengths are off when SVE is.
            _ => {
                let vq = sve_vq(prop).expect("an sve<N> property");
                self.sve && self.sve_map & (1 << (vq - 1)) != 0
            }
        }
    }

    /// Applies one `-cpu` property, which QEMU makes a global property of the type. The
    /// properties were checked when the CPU was made, so the ones that do not change what
    /// the expansion reports are passed over.
    fn apply_global(&mut self, prop: &str, value: &str) {
        let on = match value {
            "on" | "yes" | "true" => true,
            "off" | "no" | "false" => false,
            _ => {
                if let ("sve-max-vq", Ok(vq)) = (prop, value.parse::<u32>()) {
                    self.set_sve_max_vq(vq);
                }
                return;
            }
        };
        if ADVERTISED.contains(&prop) && self.has(prop) {
            let _ = self.set(prop, on);
        }
    }
}

/// The `prop` and value of one `-cpu` feature: `prop=value`, `+prop` or `-prop`.
pub(crate) fn cpu_feature(feat: &str) -> (&str, &str) {
    match feat.split_once('=') {
        Some((p, v)) => (p, v),
        None => match feat.strip_prefix('-') {
            Some(p) => (p, "off"),
            None => (feat.strip_prefix('+').unwrap_or(feat), "on"),
        },
    }
}

/// `qmp_query_cpu_model_expansion()` for TCG and qtest. `cpu` is the `-cpu` option, whose
/// properties apply to new objects of its type.
pub(crate) fn query_cpu_model_expansion(
    type_: CpuModelExpansionType,
    model: CpuModelInfo,
    cpu: Option<&str>,
    tcg: bool,
) -> Result<CpuModelExpansionInfo> {
    if type_ != CpuModelExpansionType::Full {
        return Err(Error::generic("The requested expansion type is not supported"));
    }
    let name = model.name;
    if name == "host" {
        return Err(Error::generic(format!("The CPU type '{name}' requires KVM")));
    }
    let Some(mut obj) = ArmCpuProps::new(&name, tcg) else {
        return Err(Error::generic(format!(
            "The CPU type '{name}' is not a recognized ARM CPU type"
        )));
    };
    if let Some(cpu) = cpu {
        let mut parts = cpu.split(',');
        if parts.next() == Some(name.as_str()) {
            for (prop, value) in parts.filter(|f| !f.is_empty()).map(cpu_feature) {
                obj.apply_global(prop, value);
            }
        }
    }
    match model.props {
        None => obj.finalize()?,
        Some(QValue::Dict(props)) => {
            for &prop in ADVERTISED {
                let Some(value) = props.get(prop) else { continue };
                obj.check(prop)?;
                let QValue::Bool(value) = value else {
                    return Err(Error::generic(format!(
                        "Invalid parameter type for 'model.props.{prop}', expected: boolean"
                    )));
                };
                obj.set(prop, *value)?;
            }
            // visit_check_struct()
            if let Some(key) = props.keys().find(|k| !ADVERTISED.contains(k)) {
                return Err(Error::generic(format!("Parameter 'model.props.{key}' is unexpected")));
            }
            obj.finalize()?;
        }
        Some(_) => {
            return Err(Error::generic(
                "Invalid parameter type for 'model.props', expected: object",
            ));
        }
    }
    let mut out = QDict::new();
    for &prop in ADVERTISED.iter().filter(|p| obj.has(p)) {
        out.put(prop, obj.get(prop));
    }
    let props = (!out.is_empty()).then_some(QValue::Dict(out));
    Ok(CpuModelExpansionInfo { model: CpuModelInfo { name, props }, deprecated_props: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vector lengths `max` gets with `props` set in the expansion's order, or the error.
    fn vls(cpu: Option<&str>, props: &[(&str, bool)]) -> std::result::Result<u32, String> {
        let mut d = QDict::new();
        for &(k, v) in props {
            d.put(k, v);
        }
        let model = CpuModelInfo { name: "max".into(), props: Some(QValue::Dict(d)) };
        let info = query_cpu_model_expansion(CpuModelExpansionType::Full, model, cpu, true)
            .map_err(|e| e.message().to_string())?;
        let Some(QValue::Dict(out)) = info.model.props else { panic!("no props") };
        Ok((1..=16)
            .filter(|vq| out.get(&format!("sve{}", vq * 128)) == Some(&QValue::Bool(true)))
            .map(|vq| 1 << (vq - 1))
            .sum())
    }

    #[test]
    fn sve_lengths_like_qemu() {
        assert_eq!(vls(None, &[]), Ok(0xffff));
        assert_eq!(vls(None, &[("sve", false)]), Ok(0));
        assert_eq!(vls(None, &[("sve128", false)]), Err("cannot disable sve128".into()));
        assert_eq!(vls(None, &[("sve384", true)]), Ok(0x7));
        assert_eq!(vls(None, &[("sve384", false)]), Ok(0xfffb));
        assert_eq!(vls(None, &[("sve1024", true)]), Ok(0x8b));
        assert_eq!(vls(None, &[("sve1024", true), ("sve384", true)]), Ok(0x8f));
        assert_eq!(
            vls(None, &[("sve1024", true), ("sve256", false)]),
            Err("cannot disable sve256".into())
        );
        assert_eq!(
            vls(None, &[("sve384", true), ("sve512", false), ("sve640", true)]),
            Err("cannot disable sve512".into())
        );
        assert_eq!(vls(None, &[("sve512", false)]), Ok(0x7));
        assert_eq!(vls(None, &[("sve384", true), ("sve512", true), ("sve640", false)]), Ok(0xf));

        let vq8 = Some("max,sve-max-vq=8");
        assert_eq!(vls(vq8, &[]), Ok(0xff));
        assert_eq!(vls(vq8, &[("sve1024", false)]), Err("cannot disable sve1024".into()));
        assert_eq!(vls(vq8, &[("sve1152", true)]), Err("cannot enable sve1152".into()));
        assert_eq!(vls(vq8, &[("sve384", false)]), Ok(0xfb));

        let off = Some("max,sve=off");
        assert_eq!(vls(off, &[]), Ok(0));
        assert_eq!(vls(off, &[("sve128", true)]), Err("cannot enable sve128".into()));
        assert_eq!(vls(off, &[("sve", true)]), Ok(0xffff));
        assert_eq!(vls(off, &[("sve", true), ("sve128", true), ("sve256", true)]), Ok(0x3));
        // The -cpu properties are those of max only.
        assert_eq!(vls(Some("cortex-a57,sve=off"), &[]), Ok(0xffff));
    }

    #[test]
    fn expansion_errors_like_qemu() {
        let q = |name: &str, props: Option<QValue>, tcg: bool| {
            let model = CpuModelInfo { name: name.into(), props };
            query_cpu_model_expansion(CpuModelExpansionType::Full, model, None, tcg)
                .map_err(|e| e.message().to_string())
                .map(|i| i.model.props)
        };
        let dict = |k: &str, v: QValue| Some(QValue::Dict(QDict::new().with(k, v)));
        assert_eq!(
            q("max", Some(QValue::Bool(false)), true),
            Err("Invalid parameter type for 'model.props', expected: object".into())
        );
        assert_eq!(
            q("foo", None, true),
            Err("The CPU type 'foo' is not a recognized ARM CPU type".into())
        );
        assert_eq!(q("host", None, true), Err("The CPU type 'host' requires KVM".into()));
        assert_eq!(
            q("max", dict("not-a-prop", QValue::Bool(false)), true),
            Err("Parameter 'model.props.not-a-prop' is unexpected".into())
        );
        assert_eq!(
            q("max", dict("pmu", QValue::Int(1)), true),
            Err("Invalid parameter type for 'model.props.pmu', expected: boolean".into())
        );
        assert_eq!(
            q("max", dict("kvm-steal-time", QValue::Bool(true)), true),
            Err("Property 'max-arm-cpu.kvm-steal-time' not found".into())
        );
        assert_eq!(
            q("cortex-a15", dict("sve", QValue::Bool(true)), true),
            Err("Property 'cortex-a15-arm-cpu.sve' not found".into())
        );
        assert_eq!(q("arm926", None, true), Ok(None));
        assert_eq!(
            q("cortex-a15", None, true),
            Ok(Some(QValue::Dict(QDict::new().with("pmu", true))))
        );
        assert!(q("max", dict("aarch64", QValue::Bool(false)), true).is_ok());
        assert_eq!(
            q("max", dict("aarch64", QValue::Bool(false)), false),
            Err("'aarch64' feature cannot be disabled for this accelerator".into())
        );
        assert_eq!(
            q("max", dict("sve", QValue::Bool(true)), false),
            Err("Property 'max-arm-cpu.sve' not found".into())
        );
        assert_eq!(
            q("a64fx", dict("sve384", QValue::Bool(true)), true),
            Err("cannot enable sve384".into())
        );
        let mut both = QDict::new();
        both.put("pauth", false);
        both.put("pauth-qarma3", true);
        assert_eq!(
            q("max", Some(QValue::Dict(both)), true),
            Err("cannot enable pauth-impdef, pauth-qarma3 or pauth-qarma5 without pauth".into())
        );
    }
}
