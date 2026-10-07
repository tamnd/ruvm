// SPDX-License-Identifier: GPL-2.0-or-later

//! The `serial` section of an `isa-serial` port, from hw/char/serial.c, hw/char/serial-isa.c
//! and util/fifo8.c.
//!
//! The FIFO timeout and modem status poll timers are on the virtual clock, whose value travels
//! in the `timer` section.

use std::sync::{Arc, LazyLock};

use ruvm_hw_char::serial::{Fifo8VmState, Serial, SerialVmState, UART_FIFO_LENGTH};
use ruvm_migration::SaveVm;
use ruvm_vmstate::info::Timer;
use ruvm_vmstate::{VmStateDescription, VmStateField};

/// `vmstate_fifo8`. `data` is a buffer of `capacity` bytes, 16 for the UART FIFOs.
static VMSTATE_FIFO8: LazyLock<VmStateDescription<Fifo8VmState>> = LazyLock::new(|| {
    type S = Fifo8VmState;
    VmStateDescription::new("Fifo8").version_id(1).minimum_version_id(1).fields([
        VmStateField::vbuffer("data", |_: &S| UART_FIFO_LENGTH, |s: &mut S| &mut s.data).version(1),
        VmStateField::scalar("head", |s: &mut S| &mut s.head),
        VmStateField::scalar("num", |s: &mut S| &mut s.num),
    ])
});

type S = SerialVmState;

/// `vmstate_serial_thr_ipending`.
static VMSTATE_SERIAL_THR_IPENDING: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("serial/thr_ipending")
        .version_id(1)
        .minimum_version_id(1)
        .needed(S::thr_ipending_needed)
        .field(VmStateField::scalar("thr_ipending", |s: &mut S| &mut s.thr_ipending))
});

/// `vmstate_serial_tsr`.
static VMSTATE_SERIAL_TSR: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("serial/tsr")
        .version_id(1)
        .minimum_version_id(1)
        .needed(S::tsr_needed)
        .fields([
            VmStateField::scalar("tsr_retry", |s: &mut S| &mut s.tsr_retry),
            VmStateField::scalar("thr", |s: &mut S| &mut s.thr),
            VmStateField::scalar("tsr", |s: &mut S| &mut s.tsr),
        ])
});

/// `vmstate_serial_recv_fifo`.
static VMSTATE_SERIAL_RECV_FIFO: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("serial/recv_fifo")
        .version_id(1)
        .minimum_version_id(1)
        .needed(S::recv_fifo_needed)
        .field(
            VmStateField::structure("recv_fifo", &VMSTATE_FIFO8, |s: &mut S| &mut s.recv_fifo)
                .version(1),
        )
});

/// `vmstate_serial_xmit_fifo`.
static VMSTATE_SERIAL_XMIT_FIFO: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("serial/xmit_fifo")
        .version_id(1)
        .minimum_version_id(1)
        .needed(S::xmit_fifo_needed)
        .field(
            VmStateField::structure("xmit_fifo", &VMSTATE_FIFO8, |s: &mut S| &mut s.xmit_fifo)
                .version(1),
        )
});

/// `vmstate_serial_fifo_timeout_timer`.
static VMSTATE_SERIAL_FIFO_TIMEOUT_TIMER: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("serial/fifo_timeout_timer")
        .version_id(1)
        .minimum_version_id(1)
        .needed(S::fifo_timeout_timer_needed)
        .field(VmStateField::single("fifo_timeout_timer", &Timer, |s: &mut S| {
            &mut s.fifo_timeout_timer
        }))
});

/// `vmstate_serial_timeout_ipending`.
static VMSTATE_SERIAL_TIMEOUT_IPENDING: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("serial/timeout_ipending")
        .version_id(1)
        .minimum_version_id(1)
        .needed(S::timeout_ipending_needed)
        .field(VmStateField::scalar("timeout_ipending", |s: &mut S| &mut s.timeout_ipending))
});

/// `vmstate_serial_poll`.
static VMSTATE_SERIAL_POLL: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("serial/poll")
        .version_id(1)
        .minimum_version_id(1)
        .needed(S::poll_needed)
        .fields([
            VmStateField::scalar("poll_msl", |s: &mut S| &mut s.poll_msl),
            VmStateField::single("modem_status_poll", &Timer, |s: &mut S| &mut s.modem_status_poll),
        ])
});

/// `vmstate_serial`. `serial_pre_save()` is in [`Serial::vmstate_save`]; `serial_pre_load()`
/// and the version part of `serial_post_load()` are here, the rest is
/// [`Serial::vmstate_load`].
static VMSTATE_SERIAL_STATE: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("serial")
        .version_id(3)
        .minimum_version_id(2)
        .pre_load(|s: &mut S| {
            s.thr_ipending = -1;
            s.poll_msl = -1;
            0
        })
        .post_load(|s: &mut S, version_id| {
            if version_id < 3 {
                s.fcr_vmstate = 0;
            }
            0
        })
        .fields([
            VmStateField::scalar("divider", |s: &mut S| &mut s.divider).version(2),
            VmStateField::scalar("rbr", |s: &mut S| &mut s.rbr),
            VmStateField::scalar("ier", |s: &mut S| &mut s.ier),
            VmStateField::scalar("iir", |s: &mut S| &mut s.iir),
            VmStateField::scalar("lcr", |s: &mut S| &mut s.lcr),
            VmStateField::scalar("mcr", |s: &mut S| &mut s.mcr),
            VmStateField::scalar("lsr", |s: &mut S| &mut s.lsr),
            VmStateField::scalar("msr", |s: &mut S| &mut s.msr),
            VmStateField::scalar("scr", |s: &mut S| &mut s.scr),
            VmStateField::scalar("fcr_vmstate", |s: &mut S| &mut s.fcr_vmstate).version(3),
        ])
        .subsection(&VMSTATE_SERIAL_THR_IPENDING)
        .subsection(&VMSTATE_SERIAL_TSR)
        .subsection(&VMSTATE_SERIAL_RECV_FIFO)
        .subsection(&VMSTATE_SERIAL_XMIT_FIFO)
        .subsection(&VMSTATE_SERIAL_FIFO_TIMEOUT_TIMER)
        .subsection(&VMSTATE_SERIAL_TIMEOUT_IPENDING)
        .subsection(&VMSTATE_SERIAL_POLL)
});

/// `vmstate_isa_serial`: the `SerialState` as the struct field `state`.
pub(crate) static VMSTATE_ISA_SERIAL: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("serial")
        .version_id(3)
        .minimum_version_id(2)
        .field(VmStateField::structure("state", &VMSTATE_SERIAL_STATE, |s: &mut S| s))
});

/// Registers the `serial` section of `serial`, the ISA port QEMU numbers `instance` (0 for
/// COM1), with no path prefix as in QEMU.
pub(crate) fn register(savevm: &mut SaveVm, instance: u32, serial: &Arc<Serial>) {
    let (get, put) = (Arc::clone(serial), Arc::clone(serial));
    savevm.register_vmsd(
        "",
        Some(instance),
        &VMSTATE_ISA_SERIAL,
        move || Ok(get.vmstate_save()),
        move |s| put.vmstate_load(&s),
    );
}

#[cfg(test)]
mod tests {
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    fn idle() -> S {
        S {
            divider: 0x0c,
            iir: 0x01,
            lsr: 0x60,
            msr: 0xb0,
            mcr: 0x08,
            fifo_timeout_timer: -1,
            poll_msl: -1,
            modem_status_poll: -1,
            ..S::default()
        }
    }

    #[test]
    fn serial_layout_matches_qemu() {
        let mut s = idle();
        s.fcr_vmstate = 0xc1;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_ISA_SERIAL, &mut s).unwrap();
        // divider, eight registers, fcr_vmstate and no subsections.
        assert_eq!(f.as_bytes(), &[0, 0x0c, 0, 0, 0x01, 0, 0x08, 0x60, 0xb0, 0, 0xc1]);

        // What the stream does not carry comes from serial_pre_load().
        let mut back = S { thr_ipending: 1, poll_msl: 1, ..S::default() };
        let mut b = f.into_inner();
        // What follows the section in a stream: a nested structure at the end peeks past it.
        b.push(0);
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_ISA_SERIAL, &mut back, 3).unwrap();
        assert_eq!(back.thr_ipending, -1);
        assert_eq!(back.poll_msl, -1);
        assert_eq!(back.fcr_vmstate, 0xc1);
    }

    #[test]
    fn serial_subsections() {
        let mut s = idle();
        s.recv_fifo.num = 2;
        s.recv_fifo.data[..2].copy_from_slice(b"hi");
        s.fifo_timeout_timer = 1234;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_ISA_SERIAL, &mut s).unwrap();
        let b = f.into_inner();
        let fifo_len = 2 + "serial/recv_fifo".len() + 4 + 16 + 4 + 4;
        let timer_len = 2 + "serial/fifo_timeout_timer".len() + 4 + 8;
        assert_eq!(b.len(), 11 + fifo_len + timer_len);
        let data = 11 + 2 + "serial/recv_fifo".len() + 4;
        assert_eq!(&b[data..data + 2], b"hi");
        assert_eq!(&b[b.len() - 8..], &1234i64.to_be_bytes());

        let mut back = idle();
        let mut b = b;
        b.push(0);
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_ISA_SERIAL, &mut back, 3).unwrap();
        assert_eq!(back, S { thr_ipending: -1, ..s });
    }
}
