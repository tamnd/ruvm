// SPDX-License-Identifier: GPL-2.0-or-later

//! The two 8237 style DMA controllers of a PC, QEMU's hw/dma/i8257.c. The first one does
//! byte transfers on channels 0 to 3, the second one word transfers on channels 4 to 7. A
//! device registers a transfer handler for its channel and raises DREQ; while the channel is
//! unmasked the handler is called again and again, from a timer that stands in for QEMU's
//! idle bottom half, and moves data with [`I8257::read_memory`] and [`I8257::write_memory`].
//!
//! The SB16 is the only device here that uses them, which is why they live in this crate.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use ruvm_hw_core::fw_cfg::DmaMemory;
use ruvm_hw_core::timer::{Clock, Timer};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

use crate::portio::{Portio, portio_list};

const ADDR: usize = 0;
const COUNT: usize = 1;

/// The channel of each page register port, by the low three bits of the port.
const CHANNELS: [i32; 8] = [-1, 2, 3, 1, -1, -1, -1, 0];

/// How long the idle bottom half QEMU reschedules after a run waits at most before it runs.
const IDLE_BH_NS: i64 = 10_000_000;

/// `IsaDmaTransferHandler`: gets the channel number, 0 to 7, the position in the transfer
/// and its length in bytes, and returns the new position.
pub type DmaTransferHandler = Arc<dyn Fn(u32, i32, i32) -> i32 + Send + Sync>;

/// An I/O region: its name, the port it goes at, its size and its ops.
pub type IoRegion = (&'static str, u32, u64, Arc<dyn MmioOps>);

/// `I8257Regs`.
#[derive(Clone, Copy, Debug, Default)]
struct Regs {
    now: [i32; 2],
    base: [u16; 2],
    mode: u8,
    page: u8,
    pageh: u8,
}

#[derive(Debug)]
struct State {
    regs: [Regs; 4],
    mask: u8,
    status: u8,
    flip_flop: bool,
    running: bool,
}

/// One controller, `I8257State`.
pub struct I8257 {
    dshift: u32,
    mem: Arc<dyn DmaMemory>,
    s: Mutex<State>,
    handlers: Mutex<[Option<DmaTransferHandler>; 4]>,
    dma_bh: Timer,
}

impl fmt::Debug for I8257 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("I8257").field("dshift", &self.dshift).finish_non_exhaustive()
    }
}

impl I8257 {
    /// A controller after reset. `dshift` is 0 for byte and 1 for word transfers, `mem` is
    /// guest memory and the transfers run on `clock`.
    pub fn new(dshift: u32, mem: Arc<dyn DmaMemory>, clock: &Arc<Clock>) -> Arc<I8257> {
        Arc::new_cyclic(|me: &Weak<I8257>| {
            let me = me.clone();
            let dma_bh = clock.new_timer(move || {
                if let Some(d) = me.upgrade() {
                    d.dma_run();
                }
            });
            I8257 {
                dshift,
                mem,
                s: Mutex::new(State {
                    regs: [Regs::default(); 4],
                    mask: !0,
                    status: 0,
                    flip_flop: false,
                    running: false,
                }),
                handlers: Mutex::new([None, None, None, None]),
                dma_bh,
            }
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.s.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn handlers(&self) -> MutexGuard<'_, [Option<DmaTransferHandler>; 4]> {
        self.handlers.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn write_page(&self, nport: u32, data: u32) {
        // Only the ports of a channel are registered.
        if let Ok(ichan) = usize::try_from(CHANNELS[(nport & 7) as usize]) {
            self.lock().regs[ichan].page = data as u8;
        }
    }

    fn read_page(&self, nport: u32) -> u32 {
        match usize::try_from(CHANNELS[(nport & 7) as usize]) {
            Ok(ichan) => u32::from(self.lock().regs[ichan].page),
            Err(_) => 0,
        }
    }

    /// `i8257_init_chan()`.
    fn init_chan(&self, s: &mut State, ichan: usize) {
        let r = &mut s.regs[ichan];
        r.now[ADDR] = i32::from(r.base[ADDR]) << self.dshift;
        r.now[COUNT] = 0;
    }

    /// `i8257_getff()`.
    fn getff(s: &mut State) -> bool {
        let ff = s.flip_flop;
        s.flip_flop = !ff;
        ff
    }

    fn read_chan(&self, nport: u64) -> u64 {
        let iport = (nport >> self.dshift) & 0x0f;
        let ichan = (iport >> 1) as usize;
        let nreg = iport & 1;
        let mut s = self.lock();
        let r = s.regs[ichan];
        let dir = if (r.mode >> 5) & 1 != 0 { -1 } else { 1 };
        let ff = Self::getff(&mut s);
        let val = if nreg != 0 {
            (i32::from(r.base[COUNT]) << self.dshift).wrapping_sub(r.now[COUNT])
        } else {
            r.now[ADDR].wrapping_add(r.now[COUNT].wrapping_mul(dir))
        };
        ((val >> (self.dshift + (u32::from(ff) << 3))) & 0xff) as u64
    }

    fn write_chan(&self, nport: u64, data: u64) {
        let iport = (nport >> self.dshift) & 0x0f;
        let ichan = (iport >> 1) as usize;
        let nreg = (iport & 1) as usize;
        let mut s = self.lock();
        if Self::getff(&mut s) {
            let r = &mut s.regs[ichan];
            r.base[nreg] = (r.base[nreg] & 0xff) | ((data << 8) & 0xff00) as u16;
            self.init_chan(&mut s, ichan);
        } else {
            let r = &mut s.regs[ichan];
            r.base[nreg] = (r.base[nreg] & 0xff00) | (data & 0xff) as u16;
        }
    }

    fn write_cont(&self, nport: u64, data: u64) {
        let iport = (nport >> self.dshift) & 0x0f;
        let data = data as u8;
        let mut s = self.lock();
        match iport {
            // command: QEMU keeps it only for migration, and logs the modes it does not do.
            0x00 => {}
            0x01 => {
                let ichan = data & 3;
                if data & 4 != 0 {
                    s.status |= 1 << (ichan + 4);
                } else {
                    s.status &= !(1 << (ichan + 4));
                }
                s.status &= !(1 << ichan);
                drop(s);
                self.dma_run();
            }
            // single mask
            0x02 => {
                if data & 4 != 0 {
                    s.mask |= 1 << (data & 3);
                } else {
                    s.mask &= !(1 << (data & 3));
                }
                drop(s);
                self.dma_run();
            }
            // mode
            0x03 => s.regs[usize::from(data & 3)].mode = data,
            // clear flip flop
            0x04 => s.flip_flop = false,
            // reset
            0x05 => {
                s.flip_flop = false;
                s.mask = !0;
                s.status = 0;
            }
            // clear mask for all channels
            0x06 => {
                s.mask = 0;
                drop(s);
                self.dma_run();
            }
            // write mask for all channels
            0x07 => {
                s.mask = data;
                drop(s);
                self.dma_run();
            }
            _ => {}
        }
    }

    fn read_cont(&self, nport: u64) -> u64 {
        let iport = (nport >> self.dshift) & 0x0f;
        let mut s = self.lock();
        let val = match iport {
            // status
            0x00 => {
                let v = s.status;
                s.status &= 0xf0;
                v
            }
            // mask
            0x01 => s.mask,
            _ => 0,
        };
        u64::from(val)
    }

    /// `i8257_dma_has_autoinitialization()`.
    pub fn has_autoinitialization(&self, nchan: u32) -> bool {
        (self.lock().regs[(nchan & 3) as usize].mode >> 4) & 1 != 0
    }

    /// `i8257_dma_hold_DREQ()`.
    pub fn hold_dreq(&self, nchan: u32) {
        self.lock().status |= 1 << ((nchan & 3) + 4);
        self.dma_run();
    }

    /// `i8257_dma_release_DREQ()`.
    pub fn release_dreq(&self, nchan: u32) {
        self.lock().status &= !(1 << ((nchan & 3) + 4));
        self.dma_run();
    }

    /// `i8257_channel_run()`. The handler runs without the lock, as it reads guest memory
    /// and may release its DREQ.
    fn channel_run(&self, ichan: usize) {
        let ncont = self.dshift;
        let (pos, len) = {
            let s = self.lock();
            let r = &s.regs[ichan];
            (r.now[COUNT], (i32::from(r.base[COUNT]) + 1) << ncont)
        };
        let handler = self.handlers()[ichan].clone();
        // i8257_phony_handler() for a channel nobody registered.
        let n = match handler {
            Some(h) => h(ichan as u32 + (ncont << 2), pos, len),
            None => pos,
        };
        let mut s = self.lock();
        s.regs[ichan].now[COUNT] = n;
        if n == len {
            s.status |= 1 << ichan;
        }
    }

    /// `i8257_dma_run()`: runs every unmasked channel with DREQ held, and comes back while
    /// one did.
    fn dma_run(&self) {
        let mut rearm = false;
        let mut s = self.lock();
        if s.running {
            rearm = true;
        } else {
            s.running = true;
            for ichan in 0..4 {
                let mask = 1u8 << ichan;
                if s.mask & mask == 0 && s.status & (mask << 4) != 0 {
                    drop(s);
                    self.channel_run(ichan);
                    s = self.lock();
                    rearm = true;
                }
            }
            s.running = false;
        }
        drop(s);
        if rearm && !self.dma_bh.pending() {
            // qemu_bh_schedule_idle()
            if let Some(clock) = self.dma_bh.clock() {
                self.dma_bh.modify(clock.get_ns() + IDLE_BH_NS);
            }
        }
    }

    /// `i8257_dma_register_channel()`.
    pub fn register_channel(&self, nchan: u32, handler: DmaTransferHandler) {
        self.handlers()[(nchan & 3) as usize] = Some(handler);
    }

    /// The guest address a transfer on `ichan` is at, and whether it is a verify transfer.
    fn addr(&self, nchan: u32) -> (u64, u8, bool) {
        let s = self.lock();
        let r = &s.regs[(nchan & 3) as usize];
        let addr = (i32::from(r.pageh & 0x7f) << 24) | (i32::from(r.page) << 16) | r.now[ADDR];
        (addr as u64, r.mode, r.mode & 0x0c == 0)
    }

    /// `i8257_dma_read_memory()`: fills `buf` from the transfer at `pos`. Returns the length.
    pub fn read_memory(&self, nchan: u32, buf: &mut [u8], pos: i32) -> usize {
        let len = buf.len();
        let (addr, mode, verify) = self.addr(nchan);
        if verify {
            // The device asked for data the transfer does not move. It gets zeroes.
            buf.fill(0);
            return len;
        }
        if mode & 0x20 != 0 {
            let at = addr.wrapping_sub(i64::from(pos) as u64).wrapping_sub(len as u64);
            self.mem.read(at, buf);
            // Not a reversal: the C loop only copies the back half onto the front.
            for i in 0..len >> 1 {
                buf[i] = buf[len - i - 1];
            }
        } else {
            self.mem.read(addr.wrapping_add(i64::from(pos) as u64), buf);
        }
        len
    }

    /// `i8257_dma_write_memory()`: stores `buf` into the transfer at `pos`. Returns the
    /// length.
    pub fn write_memory(&self, nchan: u32, buf: &mut [u8], pos: i32) -> usize {
        let len = buf.len();
        let (addr, mode, verify) = self.addr(nchan);
        if verify {
            return len;
        }
        if mode & 0x20 != 0 {
            let at = addr.wrapping_sub(i64::from(pos) as u64).wrapping_sub(len as u64);
            self.mem.write(at, buf);
            // QEMU then scrambles the caller's buffer the same way.
            for i in 0..len {
                buf[i] = buf[len - i - 1];
            }
        } else {
            self.mem.write(addr.wrapping_add(i64::from(pos) as u64), buf);
        }
        len
    }

    /// `i8257_reset()`: a write to the master clear register.
    pub fn reset(&self) {
        self.write_cont(0x05 << self.dshift, 0);
    }

    /// `i8257_realize()` without the address space: the I/O regions with their names, the
    /// port each one goes at, its size and ops. `pageh_base` is `None` without high page
    /// registers, as on a PC.
    pub fn io_regions(
        self: &Arc<Self>,
        base: u32,
        page_base: u32,
        pageh_base: Option<u32>,
    ) -> Vec<IoRegion> {
        let size = 8u64 << self.dshift;
        let mut out: Vec<IoRegion> =
            vec![("dma-chan", base, size, Arc::new(ChannelIo(Arc::clone(self))))];
        let page = |offset, len| Portio::<I8257> {
            offset,
            len,
            size: 1,
            read: Some(I8257::read_page),
            write: Some(I8257::write_page),
        };
        for (b, s, ops) in portio_list(self, &[page(1, 3), page(7, 1)], page_base) {
            out.push(("dma-page", b, s, ops));
        }
        if let Some(pageh_base) = pageh_base {
            let pageh = |offset, len| Portio::<I8257> {
                offset,
                len,
                size: 1,
                read: Some(I8257::read_pageh),
                write: Some(I8257::write_pageh),
            };
            for (b, s, ops) in portio_list(self, &[pageh(1, 3), pageh(7, 3)], pageh_base) {
                out.push(("dma-pageh", b, s, ops));
            }
        }
        out.push(("dma-cont", base + (8 << self.dshift), size, Arc::new(ContIo(Arc::clone(self)))));
        out
    }

    fn write_pageh(&self, nport: u32, data: u32) {
        if let Ok(ichan) = usize::try_from(CHANNELS[(nport & 7) as usize]) {
            self.lock().regs[ichan].pageh = data as u8;
        }
    }

    fn read_pageh(&self, nport: u32) -> u32 {
        match usize::try_from(CHANNELS[(nport & 7) as usize]) {
            Ok(ichan) => u32::from(self.lock().regs[ichan].pageh),
            Err(_) => 0,
        }
    }
}

/// `channel_io_ops`.
struct ChannelIo(Arc<I8257>);

impl MmioOps for ChannelIo {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(self.0.read_chan(offset))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.write_chan(offset, value);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::exact(1)
    }
}

/// `cont_io_ops`.
struct ContIo(Arc<I8257>);

impl MmioOps for ContIo {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(self.0.read_cont(offset))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.write_cont(offset, value);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::exact(1)
    }
}

/// The two controllers of a PC, `i8257_dma_init()` without high page registers: byte
/// channels at 0x00 with pages at 0x80, word channels at 0xc0 with pages at 0x88.
#[derive(Debug, Clone)]
pub struct IsaDma {
    pub low: Arc<I8257>,
    pub high: Arc<I8257>,
}

impl IsaDma {
    pub fn new(mem: Arc<dyn DmaMemory>, clock: &Arc<Clock>) -> IsaDma {
        IsaDma { low: I8257::new(0, Arc::clone(&mem), clock), high: I8257::new(1, mem, clock) }
    }

    /// `isa_bus_get_dma()`: the controller of channel `nchan`.
    pub fn get(&self, nchan: u32) -> &Arc<I8257> {
        if nchan > 3 { &self.high } else { &self.low }
    }

    /// The I/O regions of both controllers, in the order QEMU adds them.
    pub fn io_regions(&self) -> Vec<IoRegion> {
        let mut out = self.low.io_regions(0x00, 0x80, None);
        out.extend(self.high.io_regions(0xc0, 0x88, None));
        out
    }

    /// Resets both controllers.
    pub fn reset(&self) {
        self.low.reset();
        self.high.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_base::ClockType;

    struct Ram(Mutex<Vec<u8>>);

    impl DmaMemory for Ram {
        fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
            let m = self.0.lock().unwrap();
            let a = addr as usize;
            buf.copy_from_slice(&m[a..a + buf.len()]);
            true
        }

        fn write(&self, addr: u64, buf: &[u8]) -> bool {
            let mut m = self.0.lock().unwrap();
            let a = addr as usize;
            m[a..a + buf.len()].copy_from_slice(buf);
            true
        }
    }

    #[test]
    fn a_word_channel_reads_its_counter_back_and_runs_its_handler() {
        let ram: Vec<u8> = (0..=255u8).cycle().take(0x30000).collect();
        let clock = Clock::manual(ClockType::Virtual);
        let d = I8257::new(1, Arc::new(Ram(Mutex::new(ram))), &clock);
        // Channel 5: address 0x1000 words, count 0x0f words, page 2, memory to device with
        // auto-init. The offsets are within the regions, a register every two ports.
        d.write_cont(4 << 1, 0);
        d.write_chan(2 << 1, 0x00);
        d.write_chan(2 << 1, 0x10);
        d.write_chan(3 << 1, 0x0f);
        d.write_chan(3 << 1, 0x00);
        d.write_page(0x8b, 2);
        d.write_cont(3 << 1, 0x59);
        assert_eq!(d.read_page(0x8b), 2);
        assert!(d.has_autoinitialization(5));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let weak = Arc::downgrade(&d);
        let log = Arc::clone(&seen);
        d.register_channel(
            5,
            Arc::new(move |nchan, pos, len| {
                let d = weak.upgrade().unwrap();
                let mut buf = [0u8; 4];
                d.read_memory(nchan, &mut buf, pos);
                log.lock().unwrap().push((nchan, pos, len, buf));
                pos + 4
            }),
        );
        d.hold_dreq(5);
        // Still masked: nothing runs.
        assert!(seen.lock().unwrap().is_empty());
        d.write_cont(2 << 1, 1);
        // The word address 0x1000 is byte 0x2000 in page 2.
        assert_eq!(*seen.lock().unwrap(), [(5, 0, 32, [0, 1, 2, 3])]);
        clock.set_ns(IDLE_BH_NS);
        clock.run_timers();
        assert_eq!(seen.lock().unwrap()[1], (5, 4, 32, [4, 5, 6, 7]));
        // The counter reads back in words, low byte first.
        d.write_cont(4 << 1, 0);
        assert_eq!(d.read_chan(3 << 1), 0x0b);
        assert_eq!(d.read_chan(3 << 1), 0x00);
        d.release_dreq(5);
        clock.set_ns(2 * IDLE_BH_NS);
        clock.run_timers();
        assert_eq!(seen.lock().unwrap().len(), 2);
        d.reset();
        assert_eq!(d.read_cont(1 << 1), 0xff);
    }
}
