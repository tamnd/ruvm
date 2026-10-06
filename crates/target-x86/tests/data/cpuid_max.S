# SPDX-License-Identifier: GPL-2.0-or-later
# A multiboot ELF that prints CPUID for leaves 0, 0x40000000, 0x80000000 and 0xc0000000 plus 0 to
# 0x3f, subleaves 0 to 0x1f, to COM1 and exits through isa-debug-exit at 0xf4. Build it with
# as --32 -o d.o cpuid_max.S && ld -m elf_i386 -Ttext=0x100000 -o d.elf d.o and run
# qemu-system-x86_64 -M q35 -accel tcg -cpu max -kernel d.elf -display none -serial file:out.txt
# -device isa-debug-exit,iobase=0xf4,iosize=4; cpuid_max.txt keeps the leaves in range.
        .text
        .globl _start
        .align 4
        .long 0x1BADB002, 0, -0x1BADB002
_start:
        mov $0x90000, %esp
        xor %ebp, %ebp
outer:
        mov %ebp, %eax
        shl $30, %eax
        mov %eax, base
        xor %esi, %esi
leafloop:
        xor %edi, %edi
subloop:
        mov base, %eax
        add %esi, %eax
        mov %eax, leaf
        mov %edi, %ecx
        cpuid
        push %edx
        push %ecx
        push %ebx
        push %eax
        mov leaf, %eax
        call hex
        mov %edi, %eax
        call hex
        pop %eax
        call hex
        pop %eax
        call hex
        pop %eax
        call hex
        pop %eax
        call hex
        mov $10, %al
        call putc
        inc %edi
        cmp $0x20, %edi
        jb subloop
        inc %esi
        cmp $0x40, %esi
        jb leafloop
        inc %ebp
        cmp $4, %ebp
        jb outer
        mov $0xf4, %dx
        xor %eax, %eax
        out %eax, %dx
        cli
        hlt
hex:
        mov %eax, %ebx
        mov $8, %ecx
1:      rol $4, %ebx
        mov %ebx, %eax
        and $15, %eax
        mov digits(%eax), %al
        call putc
        loop 1b
        mov $32, %al
        call putc
        ret
putc:
        mov $0x3f8, %dx
        out %al, %dx
        ret
digits: .ascii "0123456789abcdef"
        .data
base:   .long 0
leaf:   .long 0
