// SPDX-License-Identifier: GPL-2.0-or-later
// Runs vector instruction bytes natively and prints the register, flag, MXCSR and memory
// results. The state is YMM0 to YMM3, MM0 and MM1, RAX, RCX and RDX, RFLAGS, MXCSR and 32
// bytes of memory that RDI points to.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
struct tc { const char *name, *code, *desc; uint8_t in[200]; uint64_t fl; uint32_t mxcsr; };
#include "vec_cases.h"
// The layout the generated code uses: offsets 0 (YMM0 to YMM3), 128 (MM0 and MM1), 144
// (RAX, RCX and RDX), 168 (RFLAGS), 176 (MXCSR) and 184 (the memory pointer).
struct st { uint8_t v[168]; uint64_t fl; uint32_t mxcsr, pad; uint64_t memp; };
static uint8_t *p;
static void e(int n, ...) { __builtin_va_list ap; __builtin_va_start(ap, n);
  for (int i = 0; i < n; i++) *p++ = (uint8_t)__builtin_va_arg(ap, int); __builtin_va_end(ap); }
static void d32(uint32_t v) { memcpy(p, &v, 4); p += 4; }
// vmovdqu ymmN, [r15 + off] or the store.
static void ymm(int n, int off, int store) { e(4, 0xc4, 0xc1, 0x7e, store ? 0x7f : 0x6f);
  e(1, 0x87 | n << 3); d32(off); }
// movq mmN, [r15 + off] or the store.
static void mmx(int n, int off, int store) { e(3, 0x41, 0x0f, store ? 0x7f : 0x6f);
  e(1, 0x87 | n << 3); d32(off); }
// mov rN, [r15 + off] or the store.
static void gpr(int n, int off, int store) { e(3, 0x49, store ? 0x89 : 0x8b, 0x87 | n << 3);
  d32(off); }
static void hex(const uint8_t *b, int n) { for (int i = 0; i < n; i++) printf("%02x", b[i]); }
int main(void) {
  uint8_t *buf = mmap(0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  static uint8_t mem[64] __attribute__((aligned(64)));
  setvbuf(stdout, NULL, _IOLBF, 0);
  printf("# SPDX-License-Identifier: GPL-2.0-or-later\n");
  printf("# name;code;input kind:seed:random mxcsr;changed chunks out;rflags out;mxcsr out\n");
  for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
    const struct tc *t = &cases[c];
    struct st s;
    memcpy(s.v, t->in, 168); memcpy(mem, t->in + 168, 32);
    s.fl = t->fl; s.mxcsr = t->mxcsr; s.memp = (uint64_t)mem;
    p = buf;
    // Save the callee saved registers, point R15 at the state and load it.
    e(10, 0x55, 0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57);
    e(3, 0x49, 0x89, 0xff);
    for (int i = 0; i < 4; i++) ymm(i, 32 * i, 0);
    for (int i = 0; i < 2; i++) mmx(i, 128 + 8 * i, 0);
    for (int i = 0; i < 3; i++) gpr(i, 144 + 8 * i, 0);
    gpr(7, 184, 0);
    e(3, 0x41, 0x0f, 0xae); e(1, 0x97); d32(176);
    e(3, 0x41, 0xff, 0xb7); d32(168); e(1, 0x9d);
    for (const char *q = t->code; *q; q += 2) { unsigned v; sscanf(q, "%2x", &v); *p++ = v; }
    e(1, 0x9c); e(3, 0x41, 0x8f, 0x87); d32(168);
    e(3, 0x41, 0x0f, 0xae); e(1, 0x9f); d32(176);
    for (int i = 0; i < 4; i++) ymm(i, 32 * i, 1);
    for (int i = 0; i < 2; i++) mmx(i, 128 + 8 * i, 1);
    for (int i = 0; i < 3; i++) gpr(i, 144 + 8 * i, 1);
    // Restore the default MXCSR, leave MMX mode and clear the upper halves.
    e(2, 0x6a, 0x00);
    e(7, 0xc7, 0x04, 0x24, 0x80, 0x1f, 0x00, 0x00);
    e(4, 0x0f, 0xae, 0x14, 0x24);
    e(1, 0x58);
    e(2, 0x0f, 0x77); e(3, 0xc5, 0xf8, 0x77);
    e(11, 0x41, 0x5f, 0x41, 0x5e, 0x41, 0x5d, 0x41, 0x5c, 0x5b, 0x5d, 0xc3);
    ((void (*)(struct st *))buf)(&s);
    printf("%s;%s;%s;", t->name, t->code, t->desc);
    // The output state as the 8-byte chunks that changed, index:value.
    uint8_t out[200];
    memcpy(out, s.v, 168); memcpy(out + 168, mem, 32);
    for (int i = 0, n = 0; i < 25; i++) {
      if (memcmp(out + 8 * i, t->in + 8 * i, 8)) {
        printf("%s%d:", n++ ? "," : "", i); hex(out + 8 * i, 8);
      }
    }
    printf(";%llx;%x\n", (unsigned long long)s.fl, s.mxcsr);
  }
  return 0;
}
