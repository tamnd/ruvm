#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Usage: python3 gen_vec.py writes vec_cases.h; build vec.c with it on an x86-64 host with
# AVX2, FMA, F16C, PCLMULQDQ, AES-NI, SHA and SSE4.2 (gcc -O1 -o vec vec.c) and run ./vec > ../tcg_vec.txt.
# Build the native harness for the MMX, SSE, AVX, AVX2, FMA and F16C test cases.
import struct
CF, PF, AF, ZF, SF, OF = 1, 4, 0x10, 0x40, 0x80, 0x800
ARITH = CF | PF | AF | ZF | SF | OF
RDI = 7
M64 = (1 << 64) - 1

# The inputs come from a splitmix64 generator seeded per case, so that the data file only
# needs the kind and the seed; tests/tcg_sse.rs has the same code. Keep the two in sync.
F32 = [0, 0x80000000, 0x3f800000, 0xbfc00000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000,
       0x7fa00000, 0x00000001, 0x807fffff, 0x7f7fffff, 0x4f000000, 0xcf000000, 0x3f000000,
       0x40200000, 0xc0200000, 0x40400000, 0x3fc00000]
F64 = [0, 0x8000000000000000, 0x3ff0000000000000, 0xbff8000000000000, 0x7ff0000000000000,
       0xfff0000000000000, 0x7ff8000000000000, 0x7ff4000000000000, 0x0000000000000001,
       0x800fffffffffffff, 0x7fefffffffffffff, 0x41e0000000000000, 0xc1e0000000000000,
       0x3fe0000000000000, 0x4004000000000000, 0xc004000000000000, 0x4008000000000000,
       0x43e0000000000000, 0x3ff8000000000000]
H16 = [0, 0x8000, 0x3c00, 0x7c00, 0xfc00, 0x7e00, 0x7d00, 0x0001, 0x03ff, 0x7bff]
W16 = [0, 0x7fff, 0x8000, 0xffff, 0x80, 0x7f]
TEXT = b'\0aabbcz\x7f\x80\xff'
MXCSR = [0x1f80, 0x1f80, 0x3f80, 0x5f80, 0x7f80, 0x9fc0]


class Rng:
    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s + 0x9e3779b97f4a7c15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & M64
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & M64
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n

    def bits(self, k):
        return self.next() >> (64 - k)

    def f32(self):
        k = self.below(10)
        if k < 3:
            return F32[self.below(len(F32))]
        if k < 5:
            return struct.unpack('<I', struct.pack('<f', float(self.below(600) - 300)))[0]
        return self.bits(1) << 31 | (100 + self.below(60)) << 23 | self.bits(23)

    def f64(self):
        k = self.below(10)
        if k < 3:
            return F64[self.below(len(F64))]
        if k < 5:
            return struct.unpack('<Q', struct.pack('<d', float(self.below(600) - 300)))[0]
        return self.bits(1) << 63 | (990 + self.below(70)) << 52 | self.bits(52)

    def val(self, kind, n):
        """n bytes of the given kind: i (random), s and d (floats), h (half floats), w
        (16-bit integers near the saturation limits), x (shift counts) or t (string bytes)."""
        b = b''
        while len(b) < n:
            if kind == 's':
                b += struct.pack('<I', self.f32())
            elif kind == 'd':
                b += struct.pack('<Q', self.f64())
            elif kind == 'h':
                k = self.below(12)
                b += struct.pack('<H', H16[k] if k < 10 else self.bits(16))
            elif kind == 'w':
                k = self.below(7)
                b += struct.pack('<H', W16[k] if k < 6 else self.bits(16))
            elif kind == 'x':
                b += struct.pack('<Q', self.below(70) if self.below(2) == 0 else self.next())
            elif kind == 't':
                b += TEXT[self.below(10):][:1]
            else:
                b += struct.pack('<Q', self.next())
        return b

    def gpr(self, kind):
        if kind == 't':
            # String lengths: small and signed, or with bit 32 set for the REX.W forms.
            if self.below(4) == 0:
                return struct.pack('<Q', 1 << 32 | self.below(20))
            return struct.pack('<Q', (self.below(41) - 20) & M64)
        if kind == 's':
            k = self.below(4)
            v = [self.next, lambda: self.bits(31), lambda: -self.bits(20) & M64, lambda: 0][k]()
            return struct.pack('<Q', v)
        return struct.pack('<Q', self.next())

    def state(self, kind, mx):
        """The input: YMM0 to YMM3, MM0 and MM1, RAX, RCX, RDX and the memory, then RFLAGS
        and MXCSR."""
        st = b''.join(self.val(kind, 32) for _ in range(4)) + self.val('i', 16)
        st += b''.join(self.gpr(kind) for _ in range(3)) + self.val(kind, 32)
        fl = 0x202 | (self.bits(12) & ARITH)
        return st, fl, MXCSR[self.below(6)] if mx else 0x1f80


cases = []


def add(name, code, kind='i', n=2, mx=None):
    for i in range(n):
        seed = len(cases) + 1
        st, fl, mxcsr = Rng(seed).state(kind, mx)
        desc = f'{kind}:{seed}:{1 if mx else 0}'
        cases.append((f'{name}_{i}', code, desc, st, fl, mxcsr))


def modrm(md, reg, rm):
    return md << 6 | (reg & 7) << 3 | (rm & 7)


def leg(name, pfx, op, form='rm', kind='i', imm=None, reg=1, rm=2, w=0, n=1, mx=None):
    """A legacy encoded instruction after 0F, with register and memory forms."""
    for mem in (False, True):
        if (mem and 'm' not in form) or (not mem and 'r' not in form):
            continue
        b = list(pfx) + ([0x48] if w else []) + [0x0f] + op
        b.append(modrm(0, reg, RDI) if mem else modrm(3, reg, rm))
        imms = imm if isinstance(imm, list) else [imm]
        for v in imms:
            c = b + ([v] if v is not None else [])
            nm = name + ('_m' if mem else '') + (f'_{v:x}' if len(imms) > 1 else '')
            add(nm, c, kind, n, mx)


def mmx(name, op, **kw):
    kw.setdefault('reg', 0)
    kw.setdefault('rm', 1)
    leg(name + '_mmx', [], op, **kw)


def vex3(mp, w, vvvv, l, pp):
    return [0xc4, 0xe0 | mp, w << 7 | ((~vvvv) & 15) << 3 | l << 2 | pp]


PP = {None: 0, 0x66: 1, 0xf3: 2, 0xf2: 3}


def vx(name, mp, pp, op, ls=(0, 1), form='rm', kind='i', imm=None, reg=1, v=2, rm=3, w=0,
       n=1, mx=None):
    """A VEX encoded instruction in map mp, for each VEX.L in ls."""
    for l in ls:
        for mem in (False, True):
            if (mem and 'm' not in form) or (not mem and 'r' not in form):
                continue
            b = vex3(mp, w, v, l, PP[pp]) + [op]
            b.append(modrm(0, reg, RDI) if mem else modrm(3, reg, rm))
            imms = imm if isinstance(imm, list) else [imm]
            for i in imms:
                c = b + ([i] if i is not None else [])
                nm = (f'{name}{"256" if l else ""}' + ('_m' if mem else '') +
                      (f'_{i:x}' if len(imms) > 1 else ''))
                add(nm, c, kind, n, mx)


# MMX and SSE2 integer operations in the 0F map, with and without 66.
INT_0F = {
    0x60: 'punpcklbw', 0x61: 'punpcklwd', 0x62: 'punpckldq', 0x63: 'packsswb',
    0x64: 'pcmpgtb', 0x65: 'pcmpgtw', 0x66: 'pcmpgtd', 0x67: 'packuswb', 0x68: 'punpckhbw',
    0x69: 'punpckhwd', 0x6a: 'punpckhdq', 0x6b: 'packssdw', 0x74: 'pcmpeqb', 0x75: 'pcmpeqw',
    0x76: 'pcmpeqd', 0xd1: 'psrlw', 0xd2: 'psrld', 0xd3: 'psrlq', 0xd4: 'paddq', 0xd5: 'pmullw',
    0xd8: 'psubusb', 0xd9: 'psubusw', 0xda: 'pminub', 0xdb: 'pand', 0xdc: 'paddusb',
    0xdd: 'paddusw', 0xde: 'pmaxub', 0xdf: 'pandn', 0xe0: 'pavgb', 0xe1: 'psraw',
    0xe2: 'psrad', 0xe3: 'pavgw', 0xe4: 'pmulhuw', 0xe5: 'pmulhw', 0xe8: 'psubsb',
    0xe9: 'psubsw', 0xea: 'pminsw', 0xeb: 'por', 0xec: 'paddsb', 0xed: 'paddsw', 0xee: 'pmaxsw',
    0xef: 'pxor', 0xf1: 'psllw', 0xf2: 'pslld', 0xf3: 'psllq', 0xf4: 'pmuludq', 0xf5: 'pmaddwd',
    0xf6: 'psadbw', 0xf8: 'psubb', 0xf9: 'psubw', 0xfa: 'psubd', 0xfb: 'psubq', 0xfc: 'paddb',
    0xfd: 'paddw', 0xfe: 'paddd',
}
for op, nm in INT_0F.items():
    kind = 'x' if nm[:3] in ('psr', 'psl') else 'w'
    mmx(nm, [op], kind=kind, n=1)
    leg(nm, [0x66], [op], kind=kind, n=1)
    vx('v' + nm, 1, 0x66, op, kind=kind, n=1)
leg('punpcklqdq', [0x66], [0x6c])
leg('punpckhqdq', [0x66], [0x6d])
vx('vpunpckhqdq', 1, 0x66, 0x6d, n=1)

# SSSE3 and SSE4 integer operations in the 0F 38 map.
SSSE3 = {0x00: 'pshufb', 0x01: 'phaddw', 0x02: 'phaddd', 0x03: 'phaddsw', 0x04: 'pmaddubsw',
         0x05: 'phsubw', 0x06: 'phsubd', 0x07: 'phsubsw', 0x08: 'psignb', 0x09: 'psignw',
         0x0a: 'psignd', 0x0b: 'pmulhrsw', 0x1c: 'pabsb', 0x1d: 'pabsw', 0x1e: 'pabsd'}
for op, nm in SSSE3.items():
    mmx(nm, [0x38, op], kind='w', n=1)
    leg(nm, [0x66], [0x38, op], kind='w', n=1)
    vx('v' + nm, 2, 0x66, op, kind='w', n=1, v=0 if op >= 0x1c else 2)
SSE41 = {0x28: 'pmuldq', 0x29: 'pcmpeqq', 0x2b: 'packusdw', 0x37: 'pcmpgtq', 0x38: 'pminsb',
         0x39: 'pminsd', 0x3a: 'pminuw', 0x3b: 'pminud', 0x3c: 'pmaxsb', 0x3d: 'pmaxsd',
         0x3e: 'pmaxuw', 0x3f: 'pmaxud', 0x40: 'pmulld'}
for op, nm in SSE41.items():
    leg(nm, [0x66], [0x38, op], kind='w', n=1)
    vx('v' + nm, 2, 0x66, op, kind='w', n=1)
for op in range(0x20, 0x26):
    for base, nm in ((0x20, 'pmovsx'), (0x30, 'pmovzx')):
        leg(f'{nm}{op - 0x20}', [0x66], [0x38, base + op - 0x20], n=1)
        vx(f'v{nm}{op - 0x20}', 2, 0x66, base + op - 0x20, v=0, n=1)
leg('phminposuw', [0x66], [0x38, 0x41], kind='w')
vx('vphminposuw', 2, 0x66, 0x41, ls=(0,), v=0)
leg('ptest', [0x66], [0x38, 0x17], n=3)
vx('vptest', 2, 0x66, 0x17, v=0, n=3)
vx('vtestps', 2, 0x66, 0x0e, v=0, n=3)
vx('vtestpd', 2, 0x66, 0x0f, v=0, n=3)
leg('pblendvb', [0x66], [0x38, 0x10])
leg('blendvps', [0x66], [0x38, 0x14])
leg('blendvpd', [0x66], [0x38, 0x15])
leg('movntdqa', [0x66], [0x38, 0x2a], form='m')

# Shifts by an immediate.
for op, sub, nm in ((0x71, 2, 'psrlw'), (0x71, 4, 'psraw'), (0x71, 6, 'psllw'),
                    (0x72, 2, 'psrld'), (0x72, 4, 'psrad'), (0x72, 6, 'pslld'),
                    (0x73, 2, 'psrlq'), (0x73, 6, 'psllq'), (0x73, 3, 'psrldq'),
                    (0x73, 7, 'pslldq')):
    imms = [1, 7, 15, 33]
    if sub not in (3, 7):
        mmx(nm + 'i', [op], form='r', reg=sub, rm=1, imm=imms, n=1)
    leg(nm + 'i', [0x66], [op], form='r', reg=sub, rm=2, imm=imms, n=1)
    vx('v' + nm + 'i', 1, 0x66, op, form='r', reg=sub, v=1, rm=2, imm=[3, 17], n=1)

# Shuffles, inserts and extracts.
mmx('pshufw', [0x70], imm=[0x1b, 0xe4], n=1)
leg('pshufd', [0x66], [0x70], imm=[0x1b, 0x93], n=1)
leg('pshufhw', [0xf3], [0x70], imm=[0x1b], n=1)
leg('pshuflw', [0xf2], [0x70], imm=[0x93], n=1)
vx('vpshufd', 1, 0x66, 0x70, v=0, imm=[0x4e], n=1)
mmx('palignr', [0x3a, 0x0f], imm=[3, 9], n=1)
leg('palignr', [0x66], [0x3a, 0x0f], imm=[3, 17], n=1)
vx('vpalignr', 3, 0x66, 0x0f, imm=[5], n=1)
mmx('pinsrw', [0xc4], reg=0, rm=0, imm=[2, 5], n=1)
leg('pinsrw', [0x66], [0xc4], reg=1, rm=0, imm=[3, 9], n=1)
mmx('pextrw', [0xc5], form='r', reg=0, rm=1, imm=[1, 6], n=1)
leg('pextrw', [0x66], [0xc5], form='r', reg=0, rm=2, imm=[5, 11], n=1)
leg('pinsrb', [0x66], [0x3a, 0x20], reg=1, rm=2, imm=[7, 31], n=1)
leg('pinsrd', [0x66], [0x3a, 0x22], reg=1, rm=1, imm=[2], n=1)
leg('pinsrq', [0x66], [0x3a, 0x22], reg=1, rm=1, imm=[1], w=1, n=1)
leg('pextrb', [0x66], [0x3a, 0x14], reg=2, rm=0, imm=[9], n=1)
leg('pextrw3a', [0x66], [0x3a, 0x15], reg=2, rm=0, imm=[3], n=1)
leg('pextrd', [0x66], [0x3a, 0x16], reg=2, rm=0, imm=[3], n=1)
leg('pextrq', [0x66], [0x3a, 0x16], reg=2, rm=0, imm=[1], w=1, n=1)
leg('extractps', [0x66], [0x3a, 0x17], reg=2, rm=0, imm=[2], n=1)
leg('insertps', [0x66], [0x3a, 0x21], imm=[0x1d, 0xb2, 0x4f], n=1, kind='s')
vx('vpinsrb', 3, 0x66, 0x20, ls=(0,), reg=1, v=2, rm=0, imm=[4], n=1)
vx('vpextrd', 3, 0x66, 0x16, ls=(0,), reg=3, v=0, rm=1, imm=[2], n=1)
vx('vinsertps', 3, 0x66, 0x21, ls=(0,), imm=[0x9c], n=1, kind='s')
vx('vpshufb', 2, 0x66, 0x00, n=1)
vx('vpermilps', 2, 0x66, 0x0c, n=1, kind='s')
vx('vpermilpd', 2, 0x66, 0x0d, n=1, kind='x')
vx('vpermilpsi', 3, 0x66, 0x04, v=0, imm=[0x1b], n=1)
vx('vpermilpdi', 3, 0x66, 0x05, v=0, imm=[0x5], n=1)
vx('vperm2f128', 3, 0x66, 0x06, ls=(1,), imm=[0x21, 0x83, 0x30], n=1)
vx('vperm2i128', 3, 0x66, 0x46, ls=(1,), imm=[0x12, 0x38], n=1)
vx('vpermq', 3, 0x66, 0x00, ls=(1,), v=0, w=1, imm=[0x1b, 0xd8], n=1)
vx('vpermpd', 3, 0x66, 0x01, ls=(1,), v=0, w=1, imm=[0x4e], n=1)
vx('vpermd', 2, 0x66, 0x36, ls=(1,), n=2)
vx('vpermps', 2, 0x66, 0x16, ls=(1,), n=2)
vx('vinsertf128', 3, 0x66, 0x18, ls=(1,), imm=[0, 1], n=1)
vx('vinserti128', 3, 0x66, 0x38, ls=(1,), imm=[1], n=1)
vx('vextractf128', 3, 0x66, 0x19, ls=(1,), reg=2, v=0, rm=1, imm=[1], n=1)
vx('vextracti128', 3, 0x66, 0x39, ls=(1,), reg=2, v=0, rm=1, imm=[0, 1], n=1)
vx('vbroadcastss', 2, 0x66, 0x18, v=0, n=1)
vx('vbroadcastsd', 2, 0x66, 0x19, ls=(1,), v=0, n=1)
vx('vbroadcastf128', 2, 0x66, 0x1a, ls=(1,), v=0, form='m', n=1)
vx('vbroadcasti128', 2, 0x66, 0x5a, ls=(1,), v=0, form='m', n=1)
vx('vpbroadcastb', 2, 0x66, 0x78, v=0, n=1)
vx('vpbroadcastw', 2, 0x66, 0x79, v=0, n=1)
vx('vpbroadcastd', 2, 0x66, 0x58, v=0, n=1)
vx('vpbroadcastq', 2, 0x66, 0x59, v=0, n=1)
vx('vpblendd', 3, 0x66, 0x02, imm=[0xa5], n=1)
vx('vpsllvd', 2, 0x66, 0x47, kind='x', n=1)
vx('vpsllvq', 2, 0x66, 0x47, kind='x', w=1, n=1)
vx('vpsrlvd', 2, 0x66, 0x45, kind='x', n=1)
vx('vpsrlvq', 2, 0x66, 0x45, kind='x', w=1, n=1)
vx('vpsravd', 2, 0x66, 0x46, kind='x', n=1)
vx('vmaskmovps', 2, 0x66, 0x2c, form='m', n=2)
vx('vmaskmovpd', 2, 0x66, 0x2d, form='m', n=1)
vx('vmaskmovps_st', 2, 0x66, 0x2e, form='m', n=2)
vx('vmaskmovpd_st', 2, 0x66, 0x2f, form='m', n=1)
vx('vpmaskmovd', 2, 0x66, 0x8c, form='m', n=1)
vx('vpmaskmovq', 2, 0x66, 0x8c, form='m', w=1, n=1)
vx('vpmaskmovd_st', 2, 0x66, 0x8e, form='m', n=1)
vx('vpmaskmovq_st', 2, 0x66, 0x8e, form='m', w=1, n=1)
vx('vblendvps', 3, 0x66, 0x4a, imm=[0x00], n=2)
vx('vblendvpd', 3, 0x66, 0x4b, imm=[0x30], n=2)
vx('vpblendvb', 3, 0x66, 0x4c, imm=[0x10], n=2)
leg('pblendw', [0x66], [0x3a, 0x0e], imm=[0x5a], n=1)
leg('blendps', [0x66], [0x3a, 0x0c], imm=[0x6], n=1)
leg('blendpd', [0x66], [0x3a, 0x0d], imm=[0x1], n=1)
vx('vblendps', 3, 0x66, 0x0c, imm=[0x93], n=1)
vx('vpblendw', 3, 0x66, 0x0e, imm=[0x3c], n=1)
leg('mpsadbw', [0x66], [0x3a, 0x42], imm=[0, 5], n=1)
vx('vmpsadbw', 3, 0x66, 0x42, imm=[0x2e], n=1)
leg('pclmulqdq', [0x66], [0x3a, 0x44], imm=[0x00, 0x01, 0x10, 0x11], n=1)
vx('vpclmulqdq', 3, 0x66, 0x44, ls=(0,), imm=[0x11], n=1)

# Moves.
for pfx, nm in ((None, 'movups'), (0x66, 'movupd'), (0xf3, 'movss'), (0xf2, 'movsd')):
    p = [pfx] if pfx else []
    leg(nm, p, [0x10], n=1)
    leg(nm + '_st', p, [0x11], form='m', n=1)
    ls = (0,) if pfx in (0xf3, 0xf2) else (0, 1)
    if pfx in (0xf3, 0xf2):
        vx('v' + nm, 1, pfx, 0x10, ls=ls, form='r', n=1)
    vx('v' + nm, 1, pfx, 0x10, ls=ls, v=0, form='m' if pfx in (0xf3, 0xf2) else 'rm', n=1)
    vx('v' + nm + '_st', 1, pfx, 0x11, ls=ls, form='m', v=0, n=1)
leg('movhlps', [], [0x12], form='r')
leg('movlps', [], [0x12], form='m')
leg('movlpd', [0x66], [0x12], form='m')
leg('movlps_st', [], [0x13], form='m')
leg('movsldup', [0xf3], [0x12])
leg('movddup', [0xf2], [0x12])
vx('vmovddup', 1, 0xf2, 0x12, v=0)
vx('vmovshdup', 1, 0xf3, 0x16, v=0)
vx('vmovhlps', 1, None, 0x12, ls=(0,), form='r')
vx('vmovlps', 1, None, 0x12, ls=(0,), form='m')
leg('unpcklps', [], [0x14])
leg('unpckhpd', [0x66], [0x15])
vx('vunpcklps', 1, None, 0x14)
vx('vunpckhpd', 1, 0x66, 0x15)
leg('movlhps', [], [0x16], form='r')
leg('movhps', [], [0x16], form='m')
leg('movhpd_st', [0x66], [0x17], form='m')
vx('vmovlhps', 1, None, 0x16, ls=(0,), form='r')
vx('vmovhpd', 1, 0x66, 0x16, ls=(0,), form='m')
leg('movaps', [], [0x28], n=1)
leg('movapd_st', [0x66], [0x29], form='m', n=1)
vx('vmovaps', 1, None, 0x28, v=0, n=1)
leg('movntps', [], [0x2b], form='m', n=1)
mmx('movq_mm', [0x6f], n=1)
mmx('movq_mm_st', [0x7f], form='m', n=1)
leg('movdqa', [0x66], [0x6f], n=1)
leg('movdqu', [0xf3], [0x6f], n=1)
leg('movdqu_st', [0xf3], [0x7f], form='m', n=1)
vx('vmovdqu', 1, 0xf3, 0x6f, v=0, n=1)
vx('vmovdqa_st', 1, 0x66, 0x7f, v=0, form='m', n=1)
mmx('movd_to_mm', [0x6e], reg=0, rm=1, n=1)
mmx('movq_to_mm', [0x6e], reg=0, rm=1, w=1, n=1)
leg('movd_to_xmm', [0x66], [0x6e], reg=1, rm=2, n=1)
leg('movq_to_xmm', [0x66], [0x6e], reg=1, rm=2, w=1, n=1)
mmx('movd_from_mm', [0x7e], reg=1, rm=0, n=1)
leg('movd_from_xmm', [0x66], [0x7e], reg=1, rm=2, n=1)
leg('movq_from_xmm', [0x66], [0x7e], reg=1, rm=2, w=1, n=1)
leg('movq_xmm', [0xf3], [0x7e], n=1)
leg('movq_xmm_st', [0x66], [0xd6], n=1)
leg('movq2dq', [0xf3], [0xd6], form='r', reg=1, rm=1, n=1)
leg('movdq2q', [0xf2], [0xd6], form='r', reg=1, rm=2, n=1)
vx('vmovd_to', 1, 0x66, 0x6e, ls=(0,), v=0, rm=1, n=1)
vx('vmovq_from', 1, 0x66, 0x7e, ls=(0,), v=0, rm=2, w=1, n=1)
vx('vmovq', 1, 0xf3, 0x7e, ls=(0,), v=0, n=1)
leg('lddqu', [0xf2], [0xf0], form='m', n=1)
mmx('movntq', [0xe7], form='m', n=1)
leg('movntdq', [0x66], [0xe7], form='m', n=1)
mmx('maskmovq', [0xf7], form='r', n=2)
leg('maskmovdqu', [0x66], [0xf7], form='r', n=2)
vx('vmaskmovdqu', 1, 0x66, 0xf7, ls=(0,), v=0, form='r', n=1)
mmx('pmovmskb', [0xd7], form='r', reg=0, rm=1, n=1)
leg('pmovmskb', [0x66], [0xd7], form='r', reg=0, rm=2, n=1)
vx('vpmovmskb', 1, 0x66, 0xd7, form='r', reg=0, v=0, n=1)
leg('movmskps', [], [0x50], form='r', reg=0, n=1, kind='s')
leg('movmskpd', [0x66], [0x50], form='r', reg=0, n=1, kind='d')
vx('vmovmskps', 1, None, 0x50, form='r', reg=0, v=0, n=1, kind='s')
add('vzeroupper', [0xc5, 0xf8, 0x77], n=1)
add('vzeroall', [0xc5, 0xfc, 0x77], n=1)
add('emms', [0x0f, 0x77], n=1)

# Floating point arithmetic: packed and scalar, single and double.
for kind, pfxs in (('s', (None, 0xf3)), ('d', (0x66, 0xf2))):
    for op, nm in ((0x58, 'add'), (0x59, 'mul'), (0x5c, 'sub'), (0x5d, 'min'), (0x5e, 'div'),
                   (0x5f, 'max'), (0x51, 'sqrt')):
        for pfx in pfxs:
            sfx = {None: 'ps', 0x66: 'pd', 0xf3: 'ss', 0xf2: 'sd'}[pfx]
            leg(nm + sfx, [pfx] if pfx else [], [op], kind=kind, n=2, mx=True)
            ls = (0,) if pfx in (0xf3, 0xf2) else (0, 1)
            vvvv = 0 if op == 0x51 and pfx in (None, 0x66) else 2
            vx('v' + nm + sfx, 1, pfx, op, ls=ls, v=vvvv, kind=kind, n=1, mx=True)
    p = [0x66] if kind == 'd' else []
    sfx = 'pd' if kind == 'd' else 'ps'
    leg('haddp' + kind, [0xf2] if kind == 's' else [0x66], [0x7c], kind=kind, mx=True)
    leg('hsubp' + kind, [0xf2] if kind == 's' else [0x66], [0x7d], kind=kind, mx=True)
    leg('addsubp' + kind, [0xf2] if kind == 's' else [0x66], [0xd0], kind=kind, mx=True)
    vx('vhaddp' + kind, 1, 0xf2 if kind == 's' else 0x66, 0x7c, kind=kind, n=1)
    vx('vaddsubp' + kind, 1, 0xf2 if kind == 's' else 0x66, 0xd0, kind=kind, n=1)
    for op, nm in ((0x54, 'and'), (0x55, 'andn'), (0x56, 'or'), (0x57, 'xor')):
        leg(nm + sfx, p, [op], kind=kind, n=1)
    vx('vandn' + sfx, 1, p[0] if p else None, 0x55, kind=kind, n=1)
    leg('shuf' + sfx, p, [0xc6], kind=kind, imm=[0x1b, 0x2], n=1)
    vx('vshuf' + sfx, 1, p[0] if p else None, 0xc6, kind=kind, imm=[0x9c], n=1)
    for pfx in ((0x66,) if kind == 'd' else (None,)) + ((0xf2,) if kind == 'd' else (0xf3,)):
        sfx2 = {None: 'ps', 0x66: 'pd', 0xf3: 'ss', 0xf2: 'sd'}[pfx]
        leg('cmp' + sfx2, [pfx] if pfx else [], [0xc2], kind=kind, imm=list(range(8)), n=1)
        ls = (0,) if pfx in (0xf3, 0xf2) else (0, 1)
        vx('vcmp' + sfx2, 1, pfx, 0xc2, ls=ls, kind=kind, imm=[0x8, 0xd, 0x13, 0x1c], n=1)
    q = [0x66] if kind == 'd' else []
    leg('ucomis' + kind, q, [0x2e], kind=kind, n=4, mx=True)
    leg('comis' + kind, q, [0x2f], kind=kind, n=4, mx=True)
    vx('vcomis' + kind, 1, q[0] if q else None, 0x2f, ls=(0,), v=0, kind=kind, n=2)
    for op, nm, ss in ((0x08, 'roundps', 's'), (0x09, 'roundpd', 'd'), (0x0a, 'roundss', 's'),
                       (0x0b, 'roundsd', 'd')):
        if ss == kind:
            leg(nm, [0x66], [0x3a, op], kind=kind, imm=[0, 1, 2, 3, 4, 9], n=1, mx=True)
    leg('dpp' + kind, [0x66], [0x3a, 0x40 if kind == 's' else 0x41], kind=kind,
        imm=[0xff, 0x31], n=1, mx=True)
for nm, l in (('vroundps', (0, 1)), ):
    vx(nm, 3, 0x66, 0x08, ls=l, v=0, kind='s', imm=[1, 0xc], n=1)
vx('vroundsd', 3, 0x66, 0x0b, ls=(0,), kind='d', imm=[2], n=1)
vx('vdpps', 3, 0x66, 0x40, ls=(1,), kind='s', imm=[0xf1], n=1)

# Conversions.
leg('cvtpi2ps', [], [0x2a], reg=1, rm=0, kind='i', n=1, mx=True)
leg('cvtpi2pd', [0x66], [0x2a], reg=1, rm=0, kind='i', n=1)
leg('cvtps2pi', [], [0x2d], reg=0, rm=2, kind='s', n=2, mx=True)
leg('cvttps2pi', [], [0x2c], reg=0, rm=2, kind='s', n=2)
leg('cvtpd2pi', [0x66], [0x2d], reg=0, rm=2, kind='d', n=2, mx=True)
leg('cvttpd2pi', [0x66], [0x2c], reg=1, rm=2, kind='d', n=2)
for pfx, nm, kind in ((0xf3, 'ss', 's'), (0xf2, 'sd', 'd')):
    for w in (0, 1):
        leg(f'cvtsi2{nm}{32 << w}', [pfx], [0x2a], reg=1, rm=0, w=w, kind='s', n=2, mx=True)
        leg(f'cvt{nm}2si{32 << w}', [pfx], [0x2d], reg=0, rm=2, w=w, kind=kind, n=3, mx=True)
        leg(f'cvtt{nm}2si{32 << w}', [pfx], [0x2c], reg=2, rm=1, w=w, kind=kind, n=3)
    vx(f'vcvtsi2{nm}', 1, pfx, 0x2a, ls=(0,), reg=1, v=2, rm=0, w=1, kind='s', n=1)
    vx(f'vcvt{nm}2si', 1, pfx, 0x2d, ls=(0,), reg=0, v=0, rm=2, kind=kind, n=1)
leg('cvtss2sd', [0xf3], [0x5a], kind='s', n=2, mx=True)
leg('cvtsd2ss', [0xf2], [0x5a], kind='d', n=2, mx=True)
leg('cvtps2pd', [], [0x5a], kind='s', n=2, mx=True)
leg('cvtpd2ps', [0x66], [0x5a], kind='d', n=2, mx=True)
leg('cvtdq2ps', [], [0x5b], kind='i', n=1, mx=True)
leg('cvtps2dq', [0x66], [0x5b], kind='s', n=2, mx=True)
leg('cvttps2dq', [0xf3], [0x5b], kind='s', n=2)
leg('cvttpd2dq', [0x66], [0xe6], kind='d', n=2)
leg('cvtdq2pd', [0xf3], [0xe6], kind='i', n=1)
leg('cvtpd2dq', [0xf2], [0xe6], kind='d', n=2, mx=True)
vx('vcvtps2pd', 1, None, 0x5a, v=0, kind='s', n=1)
vx('vcvtpd2ps', 1, 0x66, 0x5a, v=0, kind='d', n=1)
vx('vcvtdq2pd', 1, 0xf3, 0xe6, v=0, n=1)
vx('vcvttpd2dq', 1, 0x66, 0xe6, v=0, kind='d', n=1)
vx('vcvtss2sd', 1, 0xf3, 0x5a, ls=(0,), kind='s', n=1)
vx('vcvtph2ps', 2, 0x66, 0x13, v=0, kind='h', n=2)
vx('vcvtps2ph', 3, 0x66, 0x1d, reg=2, v=0, rm=1, kind='s', imm=[0, 1, 4], n=1, mx=True)

# FMA: packed and scalar forms of each operand order.
FMA = {0x96: 'vfmaddsub', 0x97: 'vfmsubadd', 0x98: 'vfmadd', 0x9a: 'vfmsub', 0x9c: 'vfnmadd',
       0x9e: 'vfnmsub'}
for base, order in ((0x00, '132'), (0x10, '213'), (0x20, '231')):
    for op, nm in FMA.items():
        for w, kind in ((0, 's'), (1, 'd')):
            vx(f'{nm}{order}p{kind}', 2, 0x66, op + base, w=w, kind=kind, n=1, mx=True)
            if op >= 0x98:
                vx(f'{nm}{order}s{kind}', 2, 0x66, op + base + 1, ls=(0,), w=w, kind=kind, n=1,
                   mx=True)

# AES-NI, SHA and the SSE4.2 string compares.
for op, nm in ((0xdc, 'aesenc'), (0xdd, 'aesenclast'), (0xde, 'aesdec'), (0xdf, 'aesdeclast')):
    leg(nm, [0x66], [0x38, op], n=2)
    vx('v' + nm, 2, 0x66, op, ls=(0,), n=1)
leg('aesimc', [0x66], [0x38, 0xdb], n=2)
vx('vaesimc', 2, 0x66, 0xdb, ls=(0,), v=0, n=1)
leg('aeskeygenassist', [0x66], [0x3a, 0xdf], imm=[0x01, 0x36, 0x8d], n=1)
vx('vaeskeygenassist', 3, 0x66, 0xdf, ls=(0,), v=0, imm=[0x1b], n=1)
leg('sha1rnds4', [], [0x3a, 0xcc], imm=[0, 1, 2, 3], n=2)
for op, nm in ((0xc8, 'sha1nexte'), (0xc9, 'sha1msg1'), (0xca, 'sha1msg2'), (0xcb, 'sha256rnds2'),
               (0xcc, 'sha256msg1'), (0xcd, 'sha256msg2')):
    leg(nm, [], [0x38, op], n=3)
PCMP = [0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x08, 0x09, 0x0c, 0x0d, 0x12, 0x14, 0x34, 0x38, 0x3a,
        0x3e, 0x40, 0x45, 0x4c, 0x59, 0x6c, 0x7f]
for op, nm in ((0x60, 'pcmpestrm'), (0x61, 'pcmpestri'), (0x62, 'pcmpistrm'),
               (0x63, 'pcmpistri')):
    leg(nm, [0x66], [0x3a, op], kind='t', imm=PCMP, n=2)
    vx('v' + nm, 3, 0x66, op, ls=(0,), v=0, kind='t', imm=[0x0c, 0x44, 0x71], n=2)
    if op < 0x62:
        leg(nm + 'w', [0x66], [0x3a, op], kind='t', imm=[0x00, 0x0d, 0x44], w=1, n=3)


def gather(name, op, w, scale, shift_op, shift):
    """An AVX2 gather from [RDI + index * scale]. A shift first makes the index register YMM2
    small enough that the elements stay in the 32 bytes of memory; YMM1 is the destination and
    YMM3 the mask."""
    for l in (0, 1):
        code = vex3(1, 0, 2, l, 1) + [shift_op, modrm(3, 2, 2), shift]
        code += vex3(2, w, 3, l, 1) + [op, modrm(0, 1, 4), scale << 6 | 2 << 3 | RDI]
        add(name + ('256' if l else ''), code, n=3)


for nm, op, w, scale, sh in (('vpgatherdd', 0x90, 0, 2, (0x72, 29)),
                             ('vpgatherdq', 0x90, 1, 3, (0x72, 30)),
                             ('vpgatherqd', 0x91, 0, 2, (0x73, 61)),
                             ('vpgatherqq', 0x91, 1, 3, (0x73, 62)),
                             ('vgatherdps', 0x92, 0, 2, (0x72, 29)),
                             ('vgatherdpd', 0x92, 1, 3, (0x72, 30)),
                             ('vgatherqps', 0x93, 0, 2, (0x73, 61)),
                             ('vgatherqpd', 0x93, 1, 3, (0x73, 62))):
    gather(nm, op, w, scale, *sh)


def h(b):
    return ''.join(f'{x:02x}' for x in b)


with open('vec_cases.h', 'w') as f:
    f.write('static const struct tc cases[] = {\n')
    for name, code, desc, st, fl, mx in cases:
        f.write('{"%s", "%s", "%s", {%s}, 0x%xull, 0x%x},\n' % (
            name, h(code), desc, ','.join(map(str, st)), fl, mx))
    f.write('};\n')
print(len(cases))
