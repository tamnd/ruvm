// SPDX-License-Identifier: GPL-2.0-or-later

//! vhost-vsock against the kernel's `/dev/vhost-vsock`. Linux only, and skipped when the device
//! is missing or cannot be opened.

#![cfg(target_os = "linux")]

mod common;

use common::{Guest, RAM_SIZE};
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::vsock::*;
use ruvm_hw_virtio::{VhostMemRegion, VirtioDeviceClass};

fn available() -> bool {
    if ruvm_vhost::kernel::VhostKernel::open(VHOST_VSOCK_PATH).is_ok() {
        return true;
    }
    eprintln!("skipping: cannot open {VHOST_VSOCK_PATH}");
    false
}

#[test]
fn cid_errors_come_first() {
    let dev = |cid| {
        let conf = VhostVsockConf { guest_cid: cid, ..VhostVsockConf::default() };
        let dev: Box<dyn VirtioDeviceClass> = Box::new(VhostVsock::new(conf));
        match Guest::try_new(false, Some(dev)) {
            Ok(_) => panic!("realize should fail"),
            Err(e) => e.to_string(),
        }
    };
    assert_eq!(dev(0), "guest-cid property must be greater than 2");
    assert_eq!(dev(u64::from(u32::MAX) + 1), "guest-cid property must be a 32-bit number");
}

#[test]
fn kernel_vsock_start_stop() {
    if !available() {
        return;
    }
    // A CID nobody else is likely to hold.
    let cid = 0x1000_0000 + u64::from(std::process::id());
    let conf = VhostVsockConf { guest_cid: cid, ..VhostVsockConf::default() };
    let dev: Box<dyn VirtioDeviceClass> = Box::new(VhostVsock::new(conf));
    let mut g = match Guest::try_new(false, Some(dev)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    assert_eq!(g.config_readq(0), cid);
    assert_ne!(g.device_features() & feature(VIRTIO_F_VERSION_1), 0);

    g.negotiate(!0);
    let _ = g.setup_queue(0, 0);
    let _ = g.setup_queue(1, 0);
    let _ = g.setup_queue(2, 0);
    // The kernel reads the rings through this process's memory. A zeroed buffer stands in for
    // guest RAM: every ring in it is empty, so the backend has nothing to do.
    let ram = vec![0u8; RAM_SIZE as usize];
    let region = VhostMemRegion {
        guest_phys_addr: 0,
        memory_size: RAM_SIZE,
        userspace_addr: ram.as_ptr() as u64,
        mmap_offset: 0,
        fd: None,
    };
    g.mmio.with_device(|_, d: &mut VhostVsock| d.set_mem_table(vec![region])).unwrap().unwrap();
    g.driver_ok();
    let started =
        g.mmio.with_device(|_, d: &mut VhostVsock| d.vhost().unwrap().is_started()).unwrap();
    assert!(started);

    g.mmio
        .with_device(|vdev, d: &mut VhostVsock| {
            d.set_status(vdev, 0).unwrap();
            assert!(!d.vhost().unwrap().is_started());
            assert_eq!(vdev.last_avail_idx(0), 0);
        })
        .unwrap();
    drop(ram);
}
