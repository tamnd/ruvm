// SPDX-License-Identifier: GPL-2.0-or-later
// Runs instruction bytes natively and prints the register, flag and memory results.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
struct tc { const char *name, *code; uint64_t r[16]; uint64_t fl; uint8_t mem[32]; uint64_t mask; };
#include "cases.h"
struct st { uint64_t r[16]; uint64_t fl; };
static uint8_t *p;
static void e(int n, ...) { __builtin_va_list ap; __builtin_va_start(ap, n);
  for (int i = 0; i < n; i++) *p++ = (uint8_t)__builtin_va_arg(ap, int); __builtin_va_end(ap); }
static void d32(uint32_t v) { memcpy(p, &v, 4); p += 4; }
static void movr(int i, int store) {
  e(3, i >= 8 ? 0x4d : 0x49, store ? 0x89 : 0x8b, 0x87 | (i & 7) << 3);
  d32(8 * i);
}
static void hex(const uint8_t *b, int n) { for (int i = 0; i < n; i++) printf("%02x", b[i]); }
int main(void) {
  uint8_t *buf = mmap(0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  static uint8_t mem[64] __attribute__((aligned(64)));
  printf("# SPDX-License-Identifier: GPL-2.0-or-later\n");
  printf("# name;code;regs in;rflags in;memory in;rflags mask;regs out;rflags out;memory out\n");
  for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
    const struct tc *t = &cases[c];
    struct st s; memcpy(s.r, t->r, sizeof(s.r)); s.fl = t->fl;
    memcpy(mem, t->mem, 32); s.r[7] = (uint64_t)mem;
    p = buf;
    e(10, 0x55, 0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57);
    e(3, 0x49, 0x89, 0xff);
    e(3, 0x41, 0xff, 0xb7); d32(128); e(1, 0x9d);
    for (int i = 0; i < 15; i++) if (i != 4) movr(i, 0);
    for (const char *q = t->code; *q; q += 2) { unsigned v; sscanf(q, "%2x", &v); *p++ = v; }
    for (int i = 0; i < 15; i++) if (i != 4) movr(i, 1);
    e(1, 0x9c); e(3, 0x41, 0x8f, 0x87); d32(128);
    e(11, 0x41, 0x5f, 0x41, 0x5e, 0x41, 0x5d, 0x41, 0x5c, 0x5b, 0x5d, 0xc3);
    ((void (*)(struct st *))buf)(&s);
    printf("%s;%s;", t->name, t->code);
    for (int i = 0; i < 16; i++) printf("%s%llx", i ? "," : "", (unsigned long long)t->r[i]);
    printf(";%llx;", (unsigned long long)t->fl); hex(t->mem, 32);
    printf(";%llx;", (unsigned long long)t->mask);
    s.r[7] -= (uint64_t)mem; // RDI out as an offset from the data, for reproducible output
    for (int i = 0; i < 16; i++) printf("%s%llx", i ? "," : "", (unsigned long long)s.r[i]);
    printf(";%llx;", (unsigned long long)s.fl); hex(mem, 32); printf("\n");
  }
  return 0;
}
