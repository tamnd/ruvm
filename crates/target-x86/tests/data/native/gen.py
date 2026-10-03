#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Usage: python3 gen.py writes cases.h; build harness.c with it on an x86-64 host with BMI2,
# ADX and MOVBE (gcc -O1 -o harness harness.c) and run ./harness > ../tcg_bmi.txt.
# Build the native harness for the BMI/ADX/MOVBE/CRC32 test cases.
import random
random.seed(1234)
CF, PF, AF, ZF, SF, OF = 1, 4, 0x10, 0x40, 0x80, 0x800
ARITH = CF | PF | AF | ZF | SF | OF
RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI = range(8)

def vex3(mp, w, vvvv, l, pp, r, rm_b=0, x=0):
    return [0xc4, ((~r >> 3) & 1) << 7 | ((~x >> 3) & 1) << 6 | ((~rm_b >> 3) & 1) << 5 | mp,
            w << 7 | ((~vvvv) & 15) << 3 | l << 2 | pp]

def modrm(md, reg, rm):
    return md << 6 | (reg & 7) << 3 | (rm & 7)

def vop(mp, pp, op, w, reg, vvvv, rm, mem=False, imm=None):
    b = vex3(mp, w, vvvv, 0, pp, reg, 0 if mem else rm) + [op]
    b.append(modrm(0, reg, RDI) if mem else modrm(3, reg, rm))
    if imm is not None:
        b.append(imm)
    return b

def legacy(pfx, op, w, reg, rm, mem=False):
    rex = 0x40 | (8 if w else 0) | (4 if reg >= 8 else 0) | (1 if (rm >= 8 and not mem) else 0)
    b = list(pfx)
    if rex != 0x40:
        b.append(rex)
    b += [0x0f, 0x38, op, modrm(0, reg, RDI) if mem else modrm(3, reg, rm)]
    return b

cases = []
def regs_rand(special=()):
    r = [0] * 16
    for i in range(16):
        k = random.randrange(5)
        r[i] = [random.getrandbits(64), random.getrandbits(32), random.getrandbits(8),
                0xffffffffffffffff, 0][k] if k < 4 else 0
    return r

def add(name, code, mask, n=6, fix=None):
    for i in range(n):
        r = regs_rand()
        if fix:
            fix(r, i)
        fl = 0x202 | (random.getrandbits(12) & ARITH)
        mem = [random.getrandbits(8) for _ in range(32)]
        cases.append((f'{name}_{i}', code, r, fl, mem, mask))

defined = {'andn': SF | ZF | OF | CF, 'bls': SF | ZF | OF | CF, 'bzhi': SF | ZF | OF | CF,
           'bextr': ZF | OF | CF}
def ctl(reg):
    def f(r, i):
        r[reg] = random.choice([random.randrange(0, 70), random.randrange(0, 70) |
                                random.randrange(0, 70) << 8, random.getrandbits(16),
                                random.getrandbits(64)])
    return f

for w in (0, 1):
    sz = 64 if w else 32
    for mem in (False, True):
        m = '_m' if mem else ''
        add(f'andn{sz}{m}', vop(2, 0, 0xf2, w, RAX, RBX, RCX, mem), defined['andn'])
        for sub, nm in ((1, 'blsr'), (2, 'blsmsk'), (3, 'blsi')):
            add(f'{nm}{sz}{m}', vop(2, 0, 0xf3, w, sub, 9, RCX, mem), defined['bls'])
        add(f'bzhi{sz}{m}', vop(2, 0, 0xf5, w, RAX, RBX, RCX, mem), defined['bzhi'],
            fix=ctl(RBX))
        add(f'pext{sz}{m}', vop(2, 2, 0xf5, w, RAX, RBX, RCX, mem), ARITH)
        add(f'pdep{sz}{m}', vop(2, 3, 0xf5, w, 10, RBX, RCX, mem), ARITH)
        add(f'mulx{sz}{m}', vop(2, 3, 0xf6, w, RAX, RBX, RCX, mem), ARITH)
        add(f'mulx_same{sz}{m}', vop(2, 3, 0xf6, w, RAX, RAX, RCX, mem), ARITH, n=2)
        add(f'bextr{sz}{m}', vop(2, 0, 0xf7, w, RAX, RBX, RCX, mem), defined['bextr'],
            fix=ctl(RBX))
        add(f'shlx{sz}{m}', vop(2, 1, 0xf7, w, RAX, RBX, RCX, mem), ARITH)
        add(f'sarx{sz}{m}', vop(2, 2, 0xf7, w, 11, RBX, RCX, mem), ARITH)
        add(f'shrx{sz}{m}', vop(2, 3, 0xf7, w, RAX, RBX, 8, mem), ARITH)
        add(f'rorx{sz}{m}', vop(3, 3, 0xf0, w, RAX, 0, RCX, mem, imm=random.randrange(256)),
            ARITH)
        add(f'adcx{sz}{m}', legacy([0x66], 0xf6, w, RAX, RBX, mem), ARITH)
        add(f'adox{sz}{m}', legacy([0xf3], 0xf6, w, 9, RBX, mem), ARITH)
        add(f'crc32{sz}{m}', legacy([0xf2], 0xf1, w, RAX, RBX, mem), ARITH)
    add(f'movbe_ld{sz}', legacy([], 0xf0, w, RAX, 0, True), ARITH)
    add(f'movbe_st{sz}', legacy([], 0xf1, w, 10, 0, True), ARITH)
add('movbe_ld16', legacy([0x66], 0xf0, 0, RAX, 0, True), ARITH)
add('movbe_st16', legacy([0x66], 0xf1, 0, RCX, 0, True), ARITH)
add('crc32_8', legacy([0xf2], 0xf0, 0, RAX, RBX), ARITH)
add('crc32_8_ah', [0xf2, 0x0f, 0x38, 0xf0, modrm(3, RAX, 4)], ARITH)
add('crc32_8_m', legacy([0xf2], 0xf0, 0, RAX, 0, True), ARITH)
add('crc32_16', legacy([0x66, 0xf2], 0xf1, 0, RAX, RBX), ARITH)
# Chains through the ADCX/ADOX carry reuse and the lazy flags of other instructions.
adcx = lambda reg, rm: legacy([0x66], 0xf6, 1, reg, rm)
adox = lambda reg, rm: legacy([0xf3], 0xf6, 1, reg, rm)
add('chain_xoxo', adcx(RAX, RBX) + adox(RCX, RDX) + adcx(8, 9) + adox(10, 11), ARITH)
add('chain_xx_oo', adcx(RAX, RBX) + adcx(8, 9) + adox(RCX, RDX) + adox(10, 11), ARITH)
add('chain_add_adcx', [0x48, 0x01, 0xd8] + adcx(RCX, RDX) + adox(8, 9), ARITH)
add('chain_sub_adox', [0x48, 0x29, 0xd8] + adox(RCX, RDX) + [0x48, 0x11, 0xd6], ARITH)
add('chain_adcx_jc', adcx(RAX, RBX) + [0x0f, 0x92, 0xc1, 0x0f, 0x90, 0xc2], ARITH)
add('chain_blsi_adc', vop(2, 0, 0xf3, 1, 3, 9, RCX) + [0x48, 0x11, 0xd8], ARITH)
add('chain_bzhi_setcc', vop(2, 0, 0xf5, 1, RAX, RBX, RCX) +
    [0x0f, 0x92, 0xc1, 0x0f, 0x90, 0xc2, 0x0f, 0x94, 0xc3], ARITH & ~(PF | AF), fix=ctl(RBX))

def h(b):
    return ''.join(f'{x:02x}' for x in b)

with open('cases.h', 'w') as f:
    f.write('static const struct tc cases[] = {\n')
    for name, code, r, fl, mem, mask in cases:
        f.write('{"%s", "%s", {%s}, 0x%xull, {%s}, 0x%xull},\n' % (
            name, h(code), ','.join('0x%xull' % v for v in r), fl, ','.join(map(str, mem)), mask))
    f.write('};\n')
print(len(cases))
