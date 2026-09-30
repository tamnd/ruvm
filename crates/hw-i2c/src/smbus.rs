// SPDX-License-Identifier: GPL-2.0-or-later

//! SMBus on top of I2C: the slave side from `hw/i2c/smbus_slave.c` and the master side from
//! `hw/i2c/smbus_master.c`.
//!
//! An SMBus slave only sees raw writes and single byte reads. It cannot tell a word write from a
//! block write of length one, so [`SmbusDevice::write_data`] gets the whole message, command byte
//! first, and works out the rest itself.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::i2c::{I2cBus, I2cEvent, I2cNak, I2cSlave};

/// `SMBUS_DATA_MAX_LEN`: a command byte, a length byte and 32 bytes of data.
pub const SMBUS_DATA_MAX_LEN: usize = 34;

/// The callbacks of `SMBusDeviceClass`. The defaults match NULL callbacks in QEMU.
pub trait SmbusDevice: Send + Sync {
    /// A quick command, a transfer with no data. `read` is the R/W bit of the address.
    fn quick_cmd(&self, read: bool) {
        let _ = read;
    }

    /// Bytes written by the master, command byte included. Never empty.
    fn write_data(&self, buf: &[u8]) {
        let _ = buf;
    }

    /// The next byte of a read. The device adds the length byte of a block read itself.
    fn receive_byte(&self) -> u8 {
        0xff
    }

    /// The legacy device reset.
    fn reset(&self) {}
}

/// The protocol state of `SMBusDevice::mode`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum SmbusMode {
    #[default]
    Idle,
    WriteData,
    ReadData,
    Done,
    Confused,
}

#[derive(Debug)]
struct SlaveState {
    mode: SmbusMode,
    data_len: usize,
    data_buf: [u8; SMBUS_DATA_MAX_LEN],
}

/// An SMBus device on an I2C bus, `SMBusDevice`: the state machine that turns I2C events into
/// [`SmbusDevice`] calls.
pub struct SmbusSlave<D> {
    dev: Arc<D>,
    state: Mutex<SlaveState>,
}

impl<D: SmbusDevice> fmt::Debug for SmbusSlave<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.lock();
        f.debug_struct("SmbusSlave").field("mode", &s.mode).field("data_len", &s.data_len).finish()
    }
}

/// What to call once the state lock is dropped.
enum Call {
    None,
    Quick(bool),
    Write(Vec<u8>),
}

impl<D: SmbusDevice> SmbusSlave<D> {
    /// Wraps `dev`.
    pub fn new(dev: Arc<D>) -> Arc<SmbusSlave<D>> {
        Arc::new(SmbusSlave {
            dev,
            state: Mutex::new(SlaveState {
                mode: SmbusMode::Idle,
                data_len: 0,
                data_buf: [0; SMBUS_DATA_MAX_LEN],
            }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, SlaveState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The device model.
    pub fn device(&self) -> &Arc<D> {
        &self.dev
    }

    /// The protocol state.
    pub fn mode(&self) -> SmbusMode {
        self.lock().mode
    }

    /// `smbus_vmstate_needed()`: a transfer is in progress.
    pub fn vmstate_needed(&self) -> bool {
        self.lock().mode != SmbusMode::Idle
    }

    fn written(s: &SlaveState) -> Call {
        Call::Write(s.data_buf[..s.data_len].to_vec())
    }

    fn run(&self, call: Call) {
        match call {
            Call::None => {}
            Call::Quick(read) => self.dev.quick_cmd(read),
            Call::Write(buf) => self.dev.write_data(&buf),
        }
    }
}

impl<D: SmbusDevice> I2cSlave for SmbusSlave<D> {
    /// `smbus_i2c_event()`.
    fn event(&self, event: I2cEvent) -> Result<(), I2cNak> {
        let mut call = Call::None;
        {
            let mut s = self.lock();
            match event {
                I2cEvent::StartSend => {
                    s.mode = match s.mode {
                        SmbusMode::Idle => SmbusMode::WriteData,
                        _ => SmbusMode::Confused,
                    };
                }
                I2cEvent::StartRecv => {
                    s.mode = match s.mode {
                        SmbusMode::Idle => SmbusMode::ReadData,
                        // A repeated start after the command byte: hand the written part to
                        // the device before the read begins.
                        SmbusMode::WriteData if s.data_len != 0 => {
                            call = Self::written(&s);
                            SmbusMode::ReadData
                        }
                        _ => SmbusMode::Confused,
                    };
                }
                I2cEvent::Finish => {
                    if s.data_len == 0 {
                        if s.mode == SmbusMode::WriteData || s.mode == SmbusMode::ReadData {
                            call = Call::Quick(s.mode == SmbusMode::ReadData);
                        }
                    } else if s.mode == SmbusMode::WriteData {
                        call = Self::written(&s);
                    }
                    s.mode = SmbusMode::Idle;
                    s.data_len = 0;
                }
                I2cEvent::Nack => {
                    s.mode = match s.mode {
                        SmbusMode::Done | SmbusMode::ReadData => SmbusMode::Done,
                        _ => SmbusMode::Confused,
                    };
                }
                I2cEvent::StartSendAsync => return Err(I2cNak),
            }
        }
        self.run(call);
        Ok(())
    }

    /// `smbus_i2c_recv()`.
    fn recv(&self) -> u8 {
        {
            let mut s = self.lock();
            if s.mode != SmbusMode::ReadData {
                s.mode = SmbusMode::Confused;
                return 0xff;
            }
        }
        self.dev.receive_byte()
    }

    /// `smbus_i2c_send()`: bytes past the buffer are dropped, but still acked.
    fn send(&self, data: u8) -> Result<(), I2cNak> {
        let mut s = self.lock();
        if s.mode == SmbusMode::WriteData && s.data_len < SMBUS_DATA_MAX_LEN {
            let n = s.data_len;
            s.data_buf[n] = data;
            s.data_len += 1;
        }
        Ok(())
    }

    fn reset(&self) {
        self.dev.reset();
    }
}

/// `smbus_quick_command()`.
pub fn smbus_quick_command(bus: &I2cBus, addr: u8, read: bool) -> Result<(), I2cNak> {
    bus.start_transfer(addr, read)?;
    bus.end_transfer();
    Ok(())
}

/// `smbus_receive_byte()`.
pub fn smbus_receive_byte(bus: &I2cBus, addr: u8) -> Result<u8, I2cNak> {
    bus.start_recv(addr)?;
    let data = bus.recv();
    bus.nack();
    bus.end_transfer();
    Ok(data)
}

/// `smbus_send_byte()`.
pub fn smbus_send_byte(bus: &I2cBus, addr: u8, data: u8) -> Result<(), I2cNak> {
    bus.start_send(addr)?;
    let _ = bus.send(data);
    bus.end_transfer();
    Ok(())
}

/// `smbus_read_byte()`.
pub fn smbus_read_byte(bus: &I2cBus, addr: u8, command: u8) -> Result<u8, I2cNak> {
    bus.start_send(addr)?;
    let _ = bus.send(command);
    if bus.start_recv(addr).is_err() {
        bus.end_transfer();
        return Err(I2cNak);
    }
    let data = bus.recv();
    bus.nack();
    bus.end_transfer();
    Ok(data)
}

/// `smbus_write_byte()`.
pub fn smbus_write_byte(bus: &I2cBus, addr: u8, command: u8, data: u8) -> Result<(), I2cNak> {
    bus.start_send(addr)?;
    let _ = bus.send(command);
    let _ = bus.send(data);
    bus.end_transfer();
    Ok(())
}

/// `smbus_read_word()`: low byte first.
pub fn smbus_read_word(bus: &I2cBus, addr: u8, command: u8) -> Result<u16, I2cNak> {
    bus.start_send(addr)?;
    let _ = bus.send(command);
    if bus.start_recv(addr).is_err() {
        bus.end_transfer();
        return Err(I2cNak);
    }
    let mut data = u16::from(bus.recv());
    data |= u16::from(bus.recv()) << 8;
    bus.nack();
    bus.end_transfer();
    Ok(data)
}

/// `smbus_write_word()`: low byte first.
pub fn smbus_write_word(bus: &I2cBus, addr: u8, command: u8, data: u16) -> Result<(), I2cNak> {
    bus.start_send(addr)?;
    let _ = bus.send(command);
    let _ = bus.send(data as u8);
    let _ = bus.send((data >> 8) as u8);
    bus.end_transfer();
    Ok(())
}

/// `smbus_read_block()`: fills `data` and returns how many bytes were read.
///
/// With `recv_len` the first byte from the slave is the length, and a length that does not fit
/// in `data` reads nothing. Without it, `data` is filled. With `send_cmd` the command byte is
/// written first, otherwise the read starts straight away.
pub fn smbus_read_block(
    bus: &I2cBus,
    addr: u8,
    command: u8,
    data: &mut [u8],
    recv_len: bool,
    send_cmd: bool,
) -> Result<usize, I2cNak> {
    if send_cmd {
        bus.start_send(addr)?;
        let _ = bus.send(command);
    }
    if bus.start_recv(addr).is_err() {
        if send_cmd {
            bus.end_transfer();
        }
        return Err(I2cNak);
    }
    let mut rlen = if recv_len { usize::from(bus.recv()) } else { data.len() };
    if rlen > data.len() {
        rlen = 0;
    }
    for b in &mut data[..rlen] {
        *b = bus.recv();
    }
    bus.nack();
    bus.end_transfer();
    Ok(rlen)
}

/// `smbus_write_block()`: at most 32 bytes of `data`, preceded by the length byte when
/// `send_len` is set.
pub fn smbus_write_block(
    bus: &I2cBus,
    addr: u8,
    command: u8,
    data: &[u8],
    send_len: bool,
) -> Result<(), I2cNak> {
    let data = &data[..data.len().min(32)];
    bus.start_send(addr)?;
    let _ = bus.send(command);
    if send_len {
        let _ = bus.send(data.len() as u8);
    }
    for &b in data {
        let _ = bus.send(b);
    }
    bus.end_transfer();
    Ok(())
}
