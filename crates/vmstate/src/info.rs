// SPDX-License-Identifier: GPL-2.0-or-later

//! Leaf types, migration/vmstate-types.c.
//!
//! A `VMStateInfo` knows how one value of a basic type goes on the wire. The integer types are all
//! big-endian, `bool` is one byte, and the `_equal` and `_le` flavours check the incoming value
//! against what the destination already holds instead of overwriting it blindly.

use ruvm_base::{Result, bail};

use crate::file::{StreamReader, StreamWriter};

/// `VMStateInfo`: how one value of type `V` is loaded and saved.
///
/// `size` is the field's element size. The integer types ignore it, the buffer types use it as the
/// number of bytes to move.
pub trait VmStateInfo<V: ?Sized>: Send + Sync {
    /// The name QEMU uses for the type in vmdesc and `-dump-vmstate` output.
    fn name(&self) -> &'static str;

    /// The `load` callback.
    fn load(&self, f: &mut StreamReader<'_>, v: &mut V, size: usize) -> Result<()>;

    /// The `save` callback.
    fn save(&self, f: &mut StreamWriter, v: &V, size: usize) -> Result<()>;
}

/// A Rust type with an obvious QEMU leaf type: `u32` is `vmstate_info_uint32`, `bool` is
/// `vmstate_info_bool` and so on.
pub trait VmStateType: Sized + 'static {
    /// The info a plain `VMSTATE_<TYPE>` field of this type uses.
    fn info() -> &'static dyn VmStateInfo<Self>;
}

macro_rules! int_info {
    ($(#[$m:meta])* $info:ident, $name:literal, $ty:ty) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $info;

        impl VmStateInfo<$ty> for $info {
            fn name(&self) -> &'static str {
                $name
            }

            fn load(&self, f: &mut StreamReader<'_>, v: &mut $ty, _size: usize) -> Result<()> {
                let mut bytes = [0u8; size_of::<$ty>()];
                for b in &mut bytes {
                    *b = f.get_byte();
                }
                *v = <$ty>::from_be_bytes(bytes);
                Ok(())
            }

            fn save(&self, f: &mut StreamWriter, v: &$ty, _size: usize) -> Result<()> {
                f.put_buffer(&v.to_be_bytes());
                Ok(())
            }
        }

        impl VmStateType for $ty {
            fn info() -> &'static dyn VmStateInfo<$ty> {
                &$info
            }
        }
    };
}

int_info!(
    /// `vmstate_info_int8`.
    Int8, "int8", i8
);
int_info!(
    /// `vmstate_info_int16`.
    Int16, "int16", i16
);
int_info!(
    /// `vmstate_info_int32`.
    Int32, "int32", i32
);
int_info!(
    /// `vmstate_info_int64`.
    Int64, "int64", i64
);
int_info!(
    /// `vmstate_info_uint8`.
    Uint8, "uint8", u8
);
int_info!(
    /// `vmstate_info_uint16`.
    Uint16, "uint16", u16
);
int_info!(
    /// `vmstate_info_uint32`.
    Uint32, "uint32", u32
);
int_info!(
    /// `vmstate_info_uint64`.
    Uint64, "uint64", u64
);

macro_rules! equal_info {
    ($(#[$m:meta])* $info:ident, $name:literal, $ty:ty, $plain:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $info;

        impl VmStateInfo<$ty> for $info {
            fn name(&self) -> &'static str {
                $name
            }

            fn load(&self, f: &mut StreamReader<'_>, v: &mut $ty, size: usize) -> Result<()> {
                let mut v2 = 0;
                $plain.load(f, &mut v2, size)?;
                if *v == v2 {
                    return Ok(());
                }
                bail!("{:x} != {:x}", *v, v2)
            }

            fn save(&self, f: &mut StreamWriter, v: &$ty, size: usize) -> Result<()> {
                $plain.save(f, v, size)
            }
        }
    };
}

equal_info!(
    /// `vmstate_info_uint8_equal`: the incoming value must match the local one.
    Uint8Equal, "uint8 equal", u8, Uint8
);
equal_info!(
    /// `vmstate_info_uint16_equal`: the incoming value must match the local one.
    Uint16Equal, "uint16 equal", u16, Uint16
);
equal_info!(
    /// `vmstate_info_int32_equal`: the incoming value must match the local one.
    Int32Equal, "int32 equal", i32, Int32
);
equal_info!(
    /// `vmstate_info_uint32_equal`: the incoming value must match the local one.
    Uint32Equal, "uint32 equal", u32, Uint32
);
equal_info!(
    /// `vmstate_info_uint64_equal`: the incoming value must match the local one. QEMU names it
    /// "int64 equal" and so does this.
    Uint64Equal, "int64 equal", u64, Uint64
);

/// `vmstate_info_int32_le`: the incoming value must be between 0 and the local value.
#[derive(Debug, Clone, Copy, Default)]
pub struct Int32Le;

impl VmStateInfo<i32> for Int32Le {
    fn name(&self) -> &'static str {
        "int32 le"
    }

    fn load(&self, f: &mut StreamReader<'_>, cur: &mut i32, size: usize) -> Result<()> {
        let mut loaded = 0;
        Int32.load(f, &mut loaded, size)?;
        if loaded >= 0 && loaded <= *cur {
            *cur = loaded;
            return Ok(());
        }
        bail!("Invalid value {loaded} expecting positive value <= {}", *cur)
    }

    fn save(&self, f: &mut StreamWriter, v: &i32, size: usize) -> Result<()> {
        Int32.save(f, v, size)
    }
}

/// `vmstate_info_bool`: one byte, and any nonzero byte loads as true.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bool;

impl VmStateInfo<bool> for Bool {
    fn name(&self) -> &'static str {
        "bool"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut bool, _size: usize) -> Result<()> {
        *v = f.get_byte() != 0;
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, v: &bool, _size: usize) -> Result<()> {
        f.put_byte(u8::from(*v));
        Ok(())
    }
}

impl VmStateType for bool {
    fn info() -> &'static dyn VmStateInfo<bool> {
        &Bool
    }
}

/// `vmstate_info_cpudouble`: a `CPU_DoubleU` as its upper and then its lower 32 bits, which is
/// the same as the 64 bit pattern in big-endian order.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuDouble;

impl VmStateInfo<u64> for CpuDouble {
    fn name(&self) -> &'static str {
        "CPU_Double_U"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut u64, _size: usize) -> Result<()> {
        let upper = u64::from(f.get_be32());
        let lower = u64::from(f.get_be32());
        *v = (upper << 32) | lower;
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, v: &u64, _size: usize) -> Result<()> {
        f.put_be32((*v >> 32) as u32);
        f.put_be32(*v as u32);
        Ok(())
    }
}

impl VmStateInfo<f64> for CpuDouble {
    fn name(&self) -> &'static str {
        "CPU_Double_U"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut f64, size: usize) -> Result<()> {
        let mut bits = 0;
        VmStateInfo::<u64>::load(self, f, &mut bits, size)?;
        *v = f64::from_bits(bits);
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, v: &f64, size: usize) -> Result<()> {
        VmStateInfo::<u64>::save(self, f, &v.to_bits(), size)
    }
}

/// `vmstate_info_buffer`: `size` raw bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct Buffer;

impl VmStateInfo<[u8]> for Buffer {
    fn name(&self) -> &'static str {
        "buffer"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut [u8], size: usize) -> Result<()> {
        let Some(dst) = v.get_mut(..size) else {
            bail!("buffer of {} bytes cannot hold {size}", v.len());
        };
        f.get_buffer(dst);
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, v: &[u8], size: usize) -> Result<()> {
        let Some(src) = v.get(..size) else {
            bail!("buffer of {} bytes cannot supply {size}", v.len());
        };
        f.put_buffer(src);
        Ok(())
    }
}

/// `vmstate_info_unused_buffer`: `size` bytes of padding for state that no longer exists. It
/// saves zeros and throws away whatever it loads.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnusedBuffer;

impl VmStateInfo<()> for UnusedBuffer {
    fn name(&self) -> &'static str {
        "unused_buffer"
    }

    fn load(&self, f: &mut StreamReader<'_>, _v: &mut (), size: usize) -> Result<()> {
        let mut buf = [0u8; 1024];
        let mut left = size;
        while left > 0 {
            let block = left.min(buf.len());
            left -= block;
            f.get_buffer(&mut buf[..block]);
        }
        Ok(())
    }

    fn save(&self, f: &mut StreamWriter, _v: &(), size: usize) -> Result<()> {
        const ZEROS: [u8; 1024] = [0; 1024];
        let mut left = size;
        while left > 0 {
            let block = left.min(ZEROS.len());
            left -= block;
            f.put_buffer(&ZEROS[..block]);
        }
        Ok(())
    }
}

/// `vmstate_info_timer`: a `QEMUTimer` goes on the wire as its expiry time in nanoseconds, -1
/// when it is not armed. Devices keep that number in their migration state and re-arm the timer
/// from it after loading.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timer;

impl VmStateInfo<i64> for Timer {
    fn name(&self) -> &'static str {
        "timer"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut i64, size: usize) -> Result<()> {
        Int64.load(f, v, size)
    }

    fn save(&self, f: &mut StreamWriter, v: &i64, size: usize) -> Result<()> {
        Int64.save(f, v, size)
    }
}
