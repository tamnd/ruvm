// SPDX-License-Identifier: GPL-2.0-or-later

//! The I2C bus core, QEMU's `hw/i2c/core.c`.
//!
//! A bus owns a list of slaves, each with a 7 bit address. A master drives a transfer by calling
//! [`I2cBus::start_send`] or [`I2cBus::start_recv`], then [`I2cBus::send`] or [`I2cBus::recv`] a
//! byte at a time, and finally [`I2cBus::end_transfer`]. Address 0 is the general call address and
//! selects every slave on the bus at once.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

/// The general call address, `I2C_BROADCAST`.
pub const I2C_BROADCAST: u8 = 0x00;

/// The slave did not acknowledge, or no slave answered the address.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct I2cNak;

impl fmt::Display for I2cNak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("i2c: no acknowledge")
    }
}

impl std::error::Error for I2cNak {}

/// What happened on the bus, `enum i2c_event`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum I2cEvent {
    /// A start (or repeated start) condition with the read bit set.
    StartRecv,
    /// A start (or repeated start) condition with the read bit clear.
    StartSend,
    /// Like [`I2cEvent::StartSend`], for a slave that answers with [`I2cSlave::send_async`].
    StartSendAsync,
    /// A stop condition.
    Finish,
    /// The master did not acknowledge the byte it just received.
    Nack,
}

/// The callbacks of `I2CSlaveClass`.
///
/// Every method has a default that matches a NULL callback in QEMU, so a model only implements
/// what it needs.
pub trait I2cSlave: Send + Sync {
    /// A start, stop or nack. An error on a start makes the transfer fail.
    fn event(&self, event: I2cEvent) -> Result<(), I2cNak> {
        let _ = event;
        Ok(())
    }

    /// The master sent a byte. A slave without a send callback never acks.
    fn send(&self, data: u8) -> Result<(), I2cNak> {
        let _ = data;
        Err(I2cNak)
    }

    /// `send_async`: like [`I2cSlave::send`], for slaves that ack later.
    fn send_async(&self, data: u8) -> Result<(), I2cNak> {
        let _ = data;
        Err(I2cNak)
    }

    /// The master reads a byte. A slave with nothing to say returns 0xff.
    fn recv(&self) -> u8 {
        0xff
    }

    /// The legacy device reset, run by [`I2cBus::reset`].
    fn reset(&self) {}
}

#[derive(Clone)]
struct Child {
    address: u8,
    dev: Arc<dyn I2cSlave>,
}

#[derive(Default)]
struct BusInner {
    /// Newest first, the order qdev keeps bus children in.
    children: Vec<Child>,
    /// `current_devs`: the slaves taking part in the transfer in progress.
    current: Vec<Arc<dyn I2cSlave>>,
    broadcast: bool,
    /// `saved_address`, as the `i2c_bus` section last brought it.
    saved_address: u8,
}

/// `vmstate_i2c_bus` (version 1).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct I2cBusVmState {
    /// The address of the transfer in progress, [`I2C_BROADCAST`] for a general call, 0xff
    /// when the bus is idle.
    pub saved_address: u8,
}

/// An I2C bus, `I2CBus`.
pub struct I2cBus {
    name: String,
    inner: Mutex<BusInner>,
}

impl fmt::Debug for I2cBus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = self.lock();
        f.debug_struct("I2cBus")
            .field("name", &self.name)
            .field("addresses", &g.children.iter().map(|c| c.address).collect::<Vec<_>>())
            .field("busy", &!g.current.is_empty())
            .field("broadcast", &g.broadcast)
            .finish()
    }
}

impl I2cBus {
    /// `i2c_init_bus()`.
    pub fn new(name: &str) -> Arc<I2cBus> {
        Arc::new(I2cBus { name: name.to_owned(), inner: Mutex::default() })
    }

    fn lock(&self) -> MutexGuard<'_, BusInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The bus name, usually `"i2c"`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Plugs `dev` in at `address`, what `i2c_slave_create_simple()` does.
    pub fn attach(&self, address: u8, dev: Arc<dyn I2cSlave>) {
        self.lock().children.insert(0, Child { address, dev });
    }

    /// Unplugs `dev`. It also leaves any transfer in progress, without seeing a stop.
    pub fn detach(&self, dev: &Arc<dyn I2cSlave>) {
        let mut g = self.lock();
        g.children.retain(|c| !Arc::ptr_eq(&c.dev, dev));
        g.current.retain(|c| !Arc::ptr_eq(c, dev));
    }

    /// `i2c_slave_set_address()`.
    pub fn set_address(&self, dev: &Arc<dyn I2cSlave>, address: u8) {
        for c in self.lock().children.iter_mut().filter(|c| Arc::ptr_eq(&c.dev, dev)) {
            c.address = address;
        }
    }

    /// The address of each slave on the bus, newest first.
    pub fn addresses(&self) -> Vec<u8> {
        self.lock().children.iter().map(|c| c.address).collect()
    }

    /// What `vmstate_i2c_bus` sends, from `i2c_bus_pre_save()`.
    pub fn vmstate_save(&self) -> I2cBusVmState {
        let g = self.lock();
        let saved_address = match g.current.first() {
            None => 0xff,
            Some(_) if g.broadcast => I2C_BROADCAST,
            Some(dev) => {
                g.children.iter().find(|c| Arc::ptr_eq(&c.dev, dev)).map_or(0xff, |c| c.address)
            }
        };
        I2cBusVmState { saved_address }
    }

    /// Loads `vmstate_i2c_bus`. As in QEMU the address is only kept: the slaves' `post_load`
    /// rejoin the transfer from it, and no slave on a q35 bus has state to migrate.
    pub fn vmstate_load(&self, v: &I2cBusVmState) {
        self.lock().saved_address = v.saved_address;
    }

    /// The `saved_address` the last `i2c_bus` section brought, for slaves that rejoin a
    /// transfer after loading.
    pub fn saved_address(&self) -> u8 {
        self.lock().saved_address
    }

    /// `i2c_bus_busy()`: a transfer is in progress.
    pub fn busy(&self) -> bool {
        !self.lock().current.is_empty()
    }

    /// Resets every slave on the bus. Like a qdev bus reset in QEMU, a transfer in progress is
    /// left alone.
    pub fn reset(&self) {
        let children = self.lock().children.clone();
        for c in children {
            c.dev.reset();
        }
    }

    /// `i2c_scan_bus()` with the default `match_and_add`: every slave at `address`, or every
    /// slave when `broadcast` is set. Without broadcast the first match wins.
    fn scan(g: &mut BusInner, address: u8, broadcast: bool) {
        for i in 0..g.children.len() {
            if g.children[i].address == address || broadcast {
                let dev = Arc::clone(&g.children[i].dev);
                g.current.insert(0, dev);
                if !broadcast {
                    return;
                }
            }
        }
    }

    /// `i2c_do_start_transfer()`.
    fn do_start_transfer(&self, address: u8, event: I2cEvent) -> Result<(), I2cNak> {
        let (current, broadcast, scanned) = {
            let mut g = self.lock();
            if address == I2C_BROADCAST {
                g.broadcast = true;
            }
            // Slaves already in the list mean this is a repeated start in the middle of a
            // transaction, which every SMBus read does, so the bus is not scanned again.
            let mut scanned = false;
            if g.current.is_empty() {
                let broadcast = g.broadcast;
                Self::scan(&mut g, address, broadcast);
                scanned = true;
            }
            if g.current.is_empty() {
                return Err(I2cNak);
            }
            (g.current.clone(), g.broadcast, scanned)
        };
        for dev in current {
            if dev.event(event).is_err() && !broadcast {
                if scanned {
                    self.end_transfer();
                }
                return Err(I2cNak);
            }
        }
        Ok(())
    }

    /// `i2c_start_transfer()`.
    pub fn start_transfer(&self, address: u8, is_recv: bool) -> Result<(), I2cNak> {
        let event = if is_recv { I2cEvent::StartRecv } else { I2cEvent::StartSend };
        self.do_start_transfer(address, event)
    }

    /// `i2c_start_recv()`.
    pub fn start_recv(&self, address: u8) -> Result<(), I2cNak> {
        self.do_start_transfer(address, I2cEvent::StartRecv)
    }

    /// `i2c_start_send()`.
    pub fn start_send(&self, address: u8) -> Result<(), I2cNak> {
        self.do_start_transfer(address, I2cEvent::StartSend)
    }

    /// `i2c_start_send_async()`.
    pub fn start_send_async(&self, address: u8) -> Result<(), I2cNak> {
        self.do_start_transfer(address, I2cEvent::StartSendAsync)
    }

    /// `i2c_end_transfer()`: a stop condition to every slave in the transfer.
    pub fn end_transfer(&self) {
        let current = {
            let mut g = self.lock();
            g.broadcast = false;
            std::mem::take(&mut g.current)
        };
        for dev in current {
            // The result of a stop is ignored, as in QEMU.
            let _ = dev.event(I2cEvent::Finish);
        }
    }

    /// `i2c_send()`: one byte to every slave in the transfer. Once a slave naks, the rest are
    /// skipped, which is what the `ret || send()` in QEMU does.
    pub fn send(&self, data: u8) -> Result<(), I2cNak> {
        let current = self.lock().current.clone();
        let mut failed = false;
        for dev in current {
            if !failed {
                failed = dev.send(data).is_err();
            }
        }
        if failed { Err(I2cNak) } else { Ok(()) }
    }

    /// `i2c_send_async()`: one byte to the first slave in the transfer.
    pub fn send_async(&self, data: u8) -> Result<(), I2cNak> {
        let first = self.lock().current.first().cloned();
        match first {
            Some(dev) => dev.send_async(data),
            None => Err(I2cNak),
        }
    }

    /// `i2c_recv()`: a byte from the addressed slave, or 0xff when there is none or the transfer
    /// is a broadcast.
    pub fn recv(&self) -> u8 {
        let first = {
            let g = self.lock();
            if g.broadcast { None } else { g.current.first().cloned() }
        };
        first.map_or(0xff, |dev| dev.recv())
    }

    /// `i2c_nack()`: the master nacks the last byte it received.
    pub fn nack(&self) {
        let current = self.lock().current.clone();
        for dev in current {
            let _ = dev.event(I2cEvent::Nack);
        }
    }
}
