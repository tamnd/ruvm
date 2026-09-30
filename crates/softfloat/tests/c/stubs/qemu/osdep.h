/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * A minimal stand in for QEMU's qemu/osdep.h, just enough to compile
 * fpu/softfloat.c on its own for the ruvm-softfloat differential harness.
 */
#ifndef RUVM_SOFTFLOAT_OSDEP_STUB_H
#define RUVM_SOFTFLOAT_OSDEP_STUB_H

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

/* The host compilers the harness supports all have __int128. */
#define CONFIG_INT128 1

#include "qemu/compiler.h"

#define g_assert_not_reached() abort()
#define g_assert(x) assert(x)

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

#endif
