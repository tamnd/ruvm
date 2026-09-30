// SPDX-License-Identifier: GPL-2.0-or-later

//! What the protocol engine needs from the machine: memory, port io, the virtual clock, device
//! lookup and GPIO wiring. The engine in system/qtest.c calls straight into the memory core, the
//! timer code and qdev. Here those calls go through [`QtestBackend`], so this crate stays below
//! the system crate that owns them.

use ruvm_base::Result;

use crate::IrqHandler;

/// The machine side of the qtest protocol. Every method stands for a QEMU call made from
/// `qtest_process_command()`, and is named after it where there is one.
pub trait QtestBackend: Send {
    /// A resolved device. The engine keeps the one it intercepted and compares later lookups
    /// against it, the way qtest.c compares `DeviceState` pointers.
    type Device: Clone + PartialEq + Send;

    /// `qtest_enabled()`. When false, `clock_step` and `clock_set` are unknown commands, as
    /// they are when qtest runs next to a real accelerator.
    fn qtest_enabled(&self) -> bool {
        true
    }

    /// `target_big_endian()`. It decides the byte order of `writew` and friends.
    fn big_endian(&self) -> bool;

    /// `address_space_read()` on the first CPU's address space.
    fn memory_read(&mut self, addr: u64, buf: &mut [u8]);

    /// `address_space_write()` on the first CPU's address space.
    fn memory_write(&mut self, addr: u64, buf: &[u8]);

    /// `cpu_inb()`, `cpu_inw()` and `cpu_inl()`, picked by `size` (1, 2 or 4). Bits above the
    /// access size are ignored.
    fn port_read(&mut self, addr: u16, size: usize) -> u32;

    /// `cpu_outb()`, `cpu_outw()` and `cpu_outl()`, picked by `size` (1, 2 or 4). The value is
    /// already truncated to the access size.
    fn port_write(&mut self, addr: u16, size: usize, value: u32);

    /// `qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL)`.
    fn clock_get_ns(&mut self) -> i64;

    /// `qemu_clock_deadline_ns_all(QEMU_CLOCK_VIRTUAL, QEMU_TIMER_ATTR_ALL)`: nanoseconds to the
    /// next timer, or a negative number when none is pending.
    fn clock_deadline_ns_all(&mut self) -> i64;

    /// `qemu_clock_advance_virtual_time()`: runs timers up to `dest` and returns the clock after.
    fn clock_advance_virtual_time(&mut self, dest: i64) -> i64;

    /// `object_resolve_path()` followed by `DEVICE()`. `None` when the path does not name a
    /// device.
    fn resolve_device(&mut self, path: &str) -> Option<Self::Device>;

    /// The inbound half of `irq_intercept_in`: `qemu_irq_set_observer()` on the input lines of
    /// every GPIO list of `dev`, with `handler` as the observer. Returns whether the device has
    /// any GPIO list at all.
    fn irq_intercept_in(&mut self, dev: &Self::Device, handler: &IrqHandler) -> bool;

    /// The outbound half of `irq_intercept_out`: for the GPIO list called `name` (`None` for the
    /// unnamed one), `qdev_intercept_gpio_out()` every output line `i` with an IRQ that calls
    /// `handler.set(i, level)`. Returns whether a list with that name exists.
    fn irq_intercept_out(
        &mut self,
        dev: &Self::Device,
        name: Option<&str>,
        handler: &IrqHandler,
    ) -> bool;

    /// `qemu_set_irq(qdev_get_gpio_in_named(dev, name, num), level)`. `name` is `None` for the
    /// unnamed list.
    fn set_irq_in(&mut self, dev: &Self::Device, name: Option<&str>, num: i32, level: i32);

    /// `module_load()`: `Ok(true)` when the module was loaded, `Ok(false)` when it was not found,
    /// and an error when loading it failed. The default loads nothing.
    fn module_load(&mut self, prefix: &str, name: &str) -> Result<bool> {
        let _ = (prefix, name);
        Ok(false)
    }

    /// The body of `qom-tests`: instantiate every class and get and set each property, reporting
    /// failures with `error_report()`. The reply is `OK` whatever happens.
    fn qom_tests(&mut self) {}

    /// Whether the `rtas` command exists, which is when the spapr machine has registered its
    /// `qtest_set_command_cb()` callback.
    fn has_rtas(&self) -> bool {
        false
    }

    /// `qtest_rtas_call()` from hw/ppc/spapr_rtas.c: runs the RTAS call called `name` and
    /// returns whether it exists (`H_SUCCESS`) or not (`H_PARAMETER`).
    fn rtas_call(&mut self, name: &str, nargs: u32, args: u64, nret: u32, rets: u64) -> bool {
        let _ = (name, nargs, args, nret, rets);
        false
    }

    /// Whether the `csr` command exists, which is when a RISC-V hart array has registered its
    /// `qtest_set_command_cb()` callback.
    fn has_csr(&self) -> bool {
        false
    }

    /// `csr_call()` from hw/riscv/riscv_hart.c. `cmd` is `get_csr` or `set_csr`, anything else
    /// leaves `val` alone. Returns the value to reply with, or `None` when the CSR access raised
    /// an exception.
    fn csr_call(&mut self, cmd: &str, cpu: u64, csrno: i32, val: u64) -> Option<u64> {
        let _ = (cmd, cpu, csrno);
        Some(val)
    }
}
