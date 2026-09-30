// SPDX-License-Identifier: GPL-2.0-or-later

//! `-device isa-debug-exit`: guest writes turn into odd exit codes.

use std::sync::{Arc, Mutex};

use ruvm_hw_misc::debugexit::{DEBUG_EXIT_DEFAULT_IOBASE, DEBUG_EXIT_DEFAULT_IOSIZE};
use ruvm_hw_misc::*;
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;

type Codes = Arc<Mutex<Vec<u64>>>;

fn setup(config: IsaDebugExitConfig) -> (Arc<AddressSpace>, Codes, IsaDebugExit) {
    let mem = Arc::new(MemorySystem::new());
    let io = mem.new_container("io", 1 << 16).unwrap();
    let io_as = mem.address_space_init(io, "I/O").unwrap();
    let codes: Codes = Arc::default();
    let c = Arc::clone(&codes);
    let dev = IsaDebugExit::realize(
        &mem,
        io,
        config,
        Arc::new(move |code| {
            c.lock().unwrap().push(code);
        }),
    )
    .unwrap();
    (io_as, codes, dev)
}

#[test]
fn defaults() {
    let config = IsaDebugExitConfig::default();
    assert_eq!(config.iobase, 0x501);
    assert_eq!(config.iosize, 2);
    assert_eq!(DEBUG_EXIT_DEFAULT_IOBASE, 0x501);
    assert_eq!(DEBUG_EXIT_DEFAULT_IOSIZE, 2);
    assert_eq!(debug_exit_code(0), 1);
    assert_eq!(debug_exit_code(0x31), 0x63);
}

#[test]
fn byte_write_exits_with_odd_code() {
    let (io, codes, dev) = setup(IsaDebugExitConfig::default());
    assert_eq!(dev.iobase(), 0x501);
    assert!(io.store(0x501, 1, 0, Endian::Little, ATTRS).is_ok());
    assert!(io.store(0x502, 1, 0x10, Endian::Little, ATTRS).is_ok());
    assert_eq!(*codes.lock().unwrap(), [1, 0x21]);
}

#[test]
fn wide_writes_and_reads() {
    let (io, codes, _) = setup(IsaDebugExitConfig::default());
    assert!(io.store(0x501, 2, 0x1234, Endian::Little, ATTRS).is_ok());
    assert_eq!(*codes.lock().unwrap(), [0x2469]);

    let (v, r) = io.load(0x501, 1, Endian::Little, ATTRS);
    assert!(r.is_ok());
    assert_eq!(v, 0);
    assert_eq!(io.load(0x501, 2, Endian::Little, ATTRS).0, 0);
}

#[test]
fn custom_base_and_size() {
    let (io, codes, dev) = setup(IsaDebugExitConfig { iobase: 0xf4, iosize: 4 });
    assert_eq!(dev.iosize(), 4);
    assert!(io.store(0xf4, 4, 0x8000_0001, Endian::Little, ATTRS).is_ok());
    assert!(io.store(0xf7, 1, 3, Endian::Little, ATTRS).is_ok());
    // Outside the window nothing happens.
    io.store(0x501, 1, 1, Endian::Little, ATTRS);
    assert_eq!(*codes.lock().unwrap(), [0x1_0000_0003, 7]);
}
