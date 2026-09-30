// SPDX-License-Identifier: GPL-2.0-or-later

//! SMBIOS tables against the fw_cfg files of real QEMU 11.1, and the `-smbios` option parser.

use std::io;
use std::path::PathBuf;

use ruvm_firmware::e820::{E820_ENTRY_SIZE, E820Entry};
use ruvm_firmware::smbios::{
    SmbiosConfig, SmbiosEntryPointType, SmbiosOptions, SmbiosPciDevice, SmbiosTopology,
    mem_array_from_e820, parse_uuid, smbios_get_tables,
};

/// CPUID leaf 1 EAX of the qemu64 model, `env->cpuid_version`: family 15, model 107, stepping 1.
const QEMU64_CPUID_VERSION: u32 = 0x0006_0fb1;
/// CPUID leaf 1 EDX of qemu64 with a single CPU.
const QEMU64_FEATURES_1CPU: u32 = 0x078b_fbfd;
/// With more than one CPU, QEMU also sets HTT (bit 28).
const QEMU64_FEATURES_SMP: u32 = 0x178b_fbfd;

const GIB: u64 = 1 << 30;

fn golden(case: &str, file: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../machine-x86/tests/data/qemu-11.1")
        .join(case)
        .join("files/etc")
        .join(file);
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// The e820 table QEMU published for the case, which is where the type 19 ranges come from.
fn e820(case: &str) -> Vec<E820Entry> {
    golden(case, "e820")
        .chunks_exact(E820_ENTRY_SIZE)
        .map(|c| E820Entry::from_bytes(c.try_into().unwrap()))
        .collect()
}

fn config(case: &str, ram_size: u64, features: u32, topology: SmbiosTopology) -> SmbiosConfig {
    SmbiosConfig {
        ram_size,
        cpuid_version: QEMU64_CPUID_VERSION,
        cpuid_features: features,
        topology,
        mem_array: mem_array_from_e820(&e820(case)),
        ..SmbiosConfig::q35()
    }
}

fn check(case: &str, cfg: &SmbiosConfig, opts: &SmbiosOptions, ep: SmbiosEntryPointType) {
    let out = smbios_get_tables(cfg, opts).unwrap();
    assert_eq!(out.tables, golden(case, "smbios/smbios-tables"), "{case} tables");
    assert_eq!(out.anchor, golden(case, "smbios/smbios-anchor"), "{case} anchor");
    assert_eq!(out.ep_type, ep);
}

#[test]
fn q35_default() {
    let cfg = config("q35", 128 << 20, QEMU64_FEATURES_1CPU, SmbiosTopology::default());
    check("q35", &cfg, &SmbiosOptions::new(), SmbiosEntryPointType::Ep32);
}

#[test]
fn q35_4g_smp4() {
    // -smp 4 on a current machine type means one socket with four cores.
    let topo = SmbiosTopology { cores: 4, ..SmbiosTopology::default() };
    let cfg = config("q35-4g-smp4", 4 * GIB, QEMU64_FEATURES_SMP, topo);
    assert_eq!(cfg.mem_array.len(), 2);
    check("q35-4g-smp4", &cfg, &SmbiosOptions::new(), SmbiosEntryPointType::Ep32);
}

#[test]
fn q35_smbios_options() {
    let mut cfg = config("q35-smbios", 20 * GIB, QEMU64_FEATURES_1CPU, SmbiosTopology::default());
    cfg.ep_type = SmbiosEntryPointType::Ep64;
    cfg.uuid = parse_uuid("12345678-9abc-def0-1234-56789abcdef0");
    let mut opts = SmbiosOptions::new();
    for arg in [
        "type=0,vendor=ACME,version=1.2.3,date=01/02/2003,release=4.5,uefi=on",
        "type=1,manufacturer=Maker,product=Prod,version=V1,serial=SER1,sku=SKU1,family=Fam",
        "type=2,manufacturer=BB,product=BP,version=BV,serial=BS,asset=BA,location=BL",
        "type=3,manufacturer=CM,version=CV,serial=CS,asset=CA,sku=CK",
        "type=4,sock_pfx=SOCK,manufacturer=PM,version=PV,serial=PS,asset=PA,part=PP,\
         max-speed=3000,current-speed=2500",
        "type=11,value=hello,value=world",
        "type=17,loc_pfx=DIMMX,bank=BANKX,manufacturer=MM,serial=MS,asset=MA,part=MP,speed=1600",
    ] {
        opts.add(arg).unwrap();
    }
    check("q35-smbios", &cfg, &opts, SmbiosEntryPointType::Ep64);
}

#[test]
fn auto_falls_back_to_64_bit_for_many_cores() {
    let topo = SmbiosTopology { cores: 300, ..SmbiosTopology::default() };
    let cfg = config("q35", 128 << 20, QEMU64_FEATURES_SMP, topo);
    let out = smbios_get_tables(&cfg, &SmbiosOptions::new()).unwrap();
    assert_eq!(out.ep_type, SmbiosEntryPointType::Ep64);
    assert_eq!(&out.anchor[..5], b"_SM3_");

    let cfg32 = SmbiosConfig { ep_type: SmbiosEntryPointType::Ep32, ..cfg };
    let err = smbios_get_tables(&cfg32, &SmbiosOptions::new()).unwrap_err();
    assert!(err.message().starts_with("SMBIOS 2.0 doesn't support number of processor"));
}

#[test]
fn auto_falls_back_to_64_bit_for_long_tables() {
    let cfg = config("q35", 128 << 20, QEMU64_FEATURES_1CPU, SmbiosTopology::default());
    let mut opts = SmbiosOptions::new();
    let long = "x".repeat(0x10000);
    opts.add(&format!("type=11,value={long}")).unwrap();
    let out = smbios_get_tables(&cfg, &opts).unwrap();
    assert_eq!(out.ep_type, SmbiosEntryPointType::Ep64);
    // QEMU frees the OEM strings during the failed 32-bit attempt, so the 64-bit tables keep
    // the count but lose the strings.
    let t11 = out.tables.windows(2).position(|w| w == [11, 5]).unwrap();
    assert_eq!(&out.tables[t11..t11 + 7], &[11, 5, 0, 0x0e, 1, 0, 0]);

    let cfg32 = SmbiosConfig { ep_type: SmbiosEntryPointType::Ep32, ..cfg };
    let err = smbios_get_tables(&cfg32, &opts).unwrap_err();
    assert!(err.message().starts_with("SMBIOS 2.1 table length "), "{err}");
    assert!(err.message().ends_with(" exceeds 65535"), "{err}");
}

#[test]
fn big_memory_uses_extended_fields() {
    let mut cfg = SmbiosConfig {
        ram_size: 3 << 40,
        cpuid_version: QEMU64_CPUID_VERSION,
        cpuid_features: QEMU64_FEATURES_1CPU,
        ..SmbiosConfig::q35()
    };
    cfg.mem_array =
        mem_array_from_e820(&[E820Entry { address: 4 << 40, length: 3 << 40, kind: 1 }]);
    let out = smbios_get_tables(&cfg, &SmbiosOptions::new()).unwrap();
    let t = &out.tables;
    // Type 16 with the capacity in the extended field.
    let t16 = t.windows(4).position(|w| w == [16, 23, 0x00, 0x10]).unwrap();
    assert_eq!(&t[t16 + 7..t16 + 11], &0x8000_0000u32.to_le_bytes());
    assert_eq!(&t[t16 + 13..t16 + 15], &2u16.to_le_bytes());
    assert_eq!(&t[t16 + 15..t16 + 23], &(3u64 << 40).to_le_bytes());
    // Two DIMMs, 2 TiB and 1 TiB, with extended sizes.
    let d0 = t.windows(4).position(|w| w == [17, 40, 0x00, 0x11]).unwrap();
    assert_eq!(&t[d0 + 12..d0 + 14], &0x7fffu16.to_le_bytes());
    assert_eq!(&t[d0 + 28..d0 + 32], &(2u32 << 20).to_le_bytes());
    let d1 = t.windows(4).position(|w| w == [17, 40, 0x01, 0x11]).unwrap();
    assert_eq!(&t[d1 + 28..d1 + 32], &(1u32 << 20).to_le_bytes());
    // A range whose KiB numbers do not fit in 32 bits uses the 64-bit addresses.
    let r = t.windows(4).position(|w| w == [19, 31, 0x00, 0x13]).unwrap();
    assert_eq!(&t[r + 4..r + 12], &[0xff; 8]);
    assert_eq!(&t[r + 15..r + 23], &(4u64 << 40).to_le_bytes());
    assert_eq!(&t[r + 23..r + 31], &((7u64 << 40) - 1).to_le_bytes());
}

#[test]
fn file_blob_goes_first_and_replaces_the_type() {
    let cfg = config("q35", 128 << 20, QEMU64_FEATURES_1CPU, SmbiosTopology::default());
    let blob = vec![1u8, 4, 0x34, 0x12, b'H', b'i', 0, 0];
    let mut opts = SmbiosOptions::new();
    let b = blob.clone();
    opts.add_with_reader("file=t1.bin", move |p| {
        assert_eq!(p, "t1.bin");
        Ok(b.clone())
    })
    .unwrap();
    let out = smbios_get_tables(&cfg, &opts).unwrap();
    assert_eq!(&out.tables[..blob.len()], &blob[..]);
    // The generated type 1 is gone, so type 3 follows the blob directly.
    assert_eq!(out.tables[blob.len()], 3);
    // Same number of structures as the plain q35 case. The largest one used to be the
    // generated type 1 (76 bytes); now it is type 4 (66 bytes).
    let plain = golden("q35", "smbios/smbios-anchor");
    assert_eq!(out.anchor[28], plain[28]);
    assert_eq!(plain[8], 76);
    assert_eq!(out.anchor[8], 66);

    let err = opts.add("type=1,serial=x").unwrap_err();
    assert_eq!(err.message(), "can't add fields, binary file already loaded!");
}

#[test]
fn type41_needs_a_root_bus_device() {
    let mut cfg = config("q35", 128 << 20, QEMU64_FEATURES_1CPU, SmbiosTopology::default());
    cfg.pci_devices.push(SmbiosPciDevice {
        id: "nic".into(),
        bus: 0,
        devfn: 0x10,
        on_root_bus: true,
    });
    cfg.pci_devices.push(SmbiosPciDevice {
        id: "deep".into(),
        bus: 3,
        devfn: 0,
        on_root_bus: false,
    });
    let mut opts = SmbiosOptions::new();
    opts.add("type=41,designation=LAN,kind=ethernet,instance=2,pcidev=nic").unwrap();
    let out = smbios_get_tables(&cfg, &opts).unwrap();
    let t = &out.tables;
    let i = t.windows(4).position(|w| w == [41, 11, 0x00, 0x29]).unwrap();
    assert_eq!(&t[i + 4..i + 11], &[1, 0x85, 2, 0, 0, 0, 0x10]);
    assert_eq!(&t[i + 11..i + 16], b"LAN\0\0");

    let mut opts = SmbiosOptions::new();
    opts.add("type=41,designation=X,pcidev=missing").unwrap();
    let err = smbios_get_tables(&cfg, &opts).unwrap_err();
    assert_eq!(err.message(), "No PCI device missing for SMBIOS type 41 entry X");

    let mut opts = SmbiosOptions::new();
    opts.add("type=41,pcidev=deep").unwrap();
    let err = smbios_get_tables(&cfg, &opts).unwrap_err();
    assert_eq!(
        err.message(),
        "Cannot create type 41 entry for PCI device deep: not attached to the root bus"
    );
}

#[test]
fn parser_accepts_good_options() {
    let mut o = SmbiosOptions::new();
    o.add("type=0,vendor=V,release=1.2,uefi=on,vm=off").unwrap();
    assert_eq!(o.type0.vendor.as_deref(), Some("V"));
    assert_eq!(o.type0.release, Some((1, 2)));
    assert!(o.type0.uefi);
    assert!(!o.type0.vm);
    // A second type=0 keeps earlier strings but resets the booleans to their defaults.
    o.add("type=0,date=d").unwrap();
    assert_eq!(o.type0.vendor.as_deref(), Some("V"));
    assert!(!o.type0.uefi);
    assert!(o.type0.vm);

    o.add("type=1,uuid=12345678-9abc-def0-1234-56789abcdef0,serial=a,,b").unwrap();
    assert_eq!(o.type1.serial.as_deref(), Some("a,b"));
    assert_eq!(o.uuid.map(|u| u[0]), Some(0x12));

    o.add("type=4,processor-family=0x10,processor-id=0x1122334455667788").unwrap();
    assert_eq!(o.type4.processor_family, 0x10);
    assert_eq!(o.type4.processor_id, 0x1122_3344_5566_7788);
    assert_eq!(o.type4.max_speed, 2000);

    o.add("type=8,internal_reference=J1,connector_type=0x12,port_type=9").unwrap();
    o.add("type=9,slot_designation=S0,slot_id=7,pci_device=ignored").unwrap();
    assert_eq!(o.type8.len(), 1);
    assert_eq!(o.type9[0].slot_id, 7);

    o.add_with_reader("type=11,value=a,path=/oem,value=c", |p| {
        assert_eq!(p, "/oem");
        Ok(b"from file".to_vec())
    })
    .unwrap();
    assert_eq!(o.type11, [b"a".to_vec(), b"from file".to_vec(), b"c".to_vec()]);

    o.add("type=17,speed=2400,bank=B").unwrap();
    assert_eq!(o.type17.speed, 2400);

    o.add("type=41,designation=D").unwrap();
    assert_eq!(o.type41[0].kind, 0x81);
    assert_eq!(o.type41[0].instance, 1);

    // id= is swallowed by QemuOpts and never reaches smbios_entry_add.
    o.add("id=foo,type=3,sku=K").unwrap();
    assert_eq!(o.type3.sku.as_deref(), Some("K"));
}

#[test]
fn parser_rejects_bad_options() {
    fn err(arg: &str) -> String {
        let mut o = SmbiosOptions::new();
        o.add_with_reader(arg, |_| Err(io::Error::from_raw_os_error(2)))
            .unwrap_err()
            .message()
            .to_owned()
    }
    assert_eq!(err("vendor=x"), "Must specify type= or file=");
    assert_eq!(err("type=128"), "out of range!");
    assert_eq!(err("type=5"), "Don't know how to build fields for SMBIOS type 5");
    assert_eq!(err("type=0,bogus=1"), "Invalid parameter 'bogus'");
    assert_eq!(err("type=0,uefi=maybe"), "Parameter 'uefi' expects 'on' or 'off'");
    assert_eq!(err("type=0,release=1"), "Invalid release");
    assert_eq!(err("type=1,uuid=nope"), "Invalid UUID");
    assert_eq!(err("type=1x"), "Parameter 'type' expects a number");
    assert_eq!(err("type=4,max-speed=fast"), "Parameter 'max-speed' expects a number");
    assert_eq!(err("type=4,max-speed=70000"), "SMBIOS CPU speed is too large (> 65535)");
    assert_eq!(err("type=9,pcidev=x"), "Invalid parameter 'pcidev'");
    assert_eq!(err("type=41,kind=modem"), "invalid parameter value: modem");
    assert_eq!(err("type=11,path=/nope"), "Could not open '/nope': No such file or directory");
    assert_eq!(err("file=/nope"), "Cannot read SMBIOS file /nope");
    assert_eq!(err("file=/x,type=1"), "Invalid parameter 'type'");

    let mut o = SmbiosOptions::new();
    let e = o.add_with_reader("type=11,path=/f", |_| Ok(b"a\0b".to_vec())).unwrap_err();
    assert_eq!(e.message(), "NUL in OEM strings value in /f");

    let mut o = SmbiosOptions::new();
    let e = o.add_with_reader("file=/short", |_| Ok(vec![1, 2])).unwrap_err();
    assert_eq!(e.message(), "Cannot read SMBIOS file /short");

    let mut o = SmbiosOptions::new();
    o.add("type=2,serial=s").unwrap();
    let e = o.add_with_reader("file=/t2", |_| Ok(vec![2, 4, 0, 0, 0, 0])).unwrap_err();
    assert_eq!(e.message(), "can't load type 2 struct, fields already specified!");
}

#[test]
fn entry_point_type_names() {
    assert_eq!(SmbiosEntryPointType::parse("32"), Some(SmbiosEntryPointType::Ep32));
    assert_eq!(SmbiosEntryPointType::parse("64"), Some(SmbiosEntryPointType::Ep64));
    assert_eq!(SmbiosEntryPointType::parse("auto"), Some(SmbiosEntryPointType::Auto));
    assert_eq!(SmbiosEntryPointType::parse("16"), None);
}
