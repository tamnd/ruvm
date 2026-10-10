// SPDX-License-Identifier: GPL-2.0-or-later

//! The bitfields of the exit context the run loop reads, and the register value a port read
//! leaves in RAX, from `whpx_handle_portio()`.

/// `WHV_VP_EXIT_CONTEXT` past the execution state: the instruction length is the low four
/// bits of the byte after it and CR8 the high four.
pub fn instruction_length(bits: u8) -> u8 {
    bits & 0xf
}

/// CR8 from the same byte.
pub fn cr8(bits: u8) -> u64 {
    u64::from(bits >> 4)
}

/// `WHV_X64_IO_PORT_ACCESS_INFO`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct IoInfo {
    /// `IsWrite`.
    pub write: bool,
    /// `AccessSize`: 1, 2 or 4.
    pub size: usize,
    /// `StringOp`: INS or OUTS.
    pub string: bool,
    /// `RepPrefix`.
    pub rep: bool,
}

impl IoInfo {
    /// Decodes `AccessInfo.AsUINT32`.
    pub fn new(bits: u32) -> IoInfo {
        IoInfo {
            write: bits & 1 != 0,
            size: ((bits >> 1) & 7) as usize,
            string: bits & 1 << 4 != 0,
            rep: bits & 1 << 5 != 0,
        }
    }
}

/// RAX after an IN of `size` bytes that read `val`: a byte or a word only replaces the low
/// part, a dword clears the high half as any 32 bit write does.
pub fn port_read_rax(rax: u64, size: usize, val: u64) -> u64 {
    match size {
        1 => (rax & !0xff) | (val & 0xff),
        2 => (rax & !0xffff) | (val & 0xffff),
        4 => val & 0xffff_ffff,
        _ => val,
    }
}

/// `WHV_VP_EXCEPTION_INFO`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExceptionInfo {
    /// `ErrorCodeValid`.
    pub error_code_valid: bool,
    /// `SoftwareException`.
    pub software: bool,
}

impl ExceptionInfo {
    /// Decodes `ExceptionInfo.AsUINT32`.
    pub fn new(bits: u32) -> ExceptionInfo {
        ExceptionInfo { error_code_valid: bits & 1 != 0, software: bits & 2 != 0 }
    }
}

/// `WHV_EMULATOR_STATUS.EmulationSuccessful`.
pub fn emulation_ok(status: u32) -> bool {
    status & 1 != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_bits() {
        assert_eq!(instruction_length(0x52), 2);
        assert_eq!(cr8(0x52), 5);
        let out_dx_al = IoInfo::new(1 | 1 << 1);
        assert_eq!(out_dx_al, IoInfo { write: true, size: 1, string: false, rep: false });
        let rep_insw = IoInfo::new(2 << 1 | 1 << 4 | 1 << 5);
        assert_eq!(rep_insw, IoInfo { write: false, size: 2, string: true, rep: true });
        assert_eq!(
            ExceptionInfo::new(1),
            ExceptionInfo { error_code_valid: true, software: false }
        );
        assert!(emulation_ok(1) && !emulation_ok(2));
    }

    #[test]
    fn port_reads() {
        let rax = 0x1122_3344_5566_7788;
        assert_eq!(port_read_rax(rax, 1, 0xab), 0x1122_3344_5566_77ab);
        assert_eq!(port_read_rax(rax, 2, 0xabcd), 0x1122_3344_5566_abcd);
        assert_eq!(port_read_rax(rax, 4, 0xdead_beef), 0xdead_beef);
    }
}
