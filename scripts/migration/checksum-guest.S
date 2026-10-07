# SPDX-License-Identifier: GPL-2.0-or-later
#
# Migration test guest: a 64 KiB -bios image. It enters flat protected mode with interrupts
# off, fills 8 MiB at 1 MiB (every fourth page left zero), then loops forever: each step adds
# a value derived from the step number to one dword and keeps a running sum in %ebp and at
# 0x8004 (the step is in %ebx and at 0x8000). Every 2^14 steps it sums the whole area and
# prints "STEP SUM OK" (or BAD) to the debugcon port 0xe9.
#
# Build: as --32 -o checksum-guest.o checksum-guest.S
#        ld -m elf_i386 -Ttext=0xffff0000 -e 0xffff0000 -o checksum-guest.elf checksum-guest.o
#        objcopy -O binary checksum-guest.elf checksum-guest.bin
        .set BASE, 0x100000
        .set NPAGES, 2048
        .set END, BASE + NPAGES * 4096
        .text
        .code32
_start:
pm_entry:
        movw $0x10, %ax
        movw %ax, %ds
        movw %ax, %es
        movw %ax, %ss
        movw %ax, %fs
        movw %ax, %gs
        movl $0x7000, %esp
        cld
        xorl %ecx, %ecx
fill:
        testl $3, %ecx
        jz next_page
        movl %ecx, %edi
        shll $12, %edi
        addl $BASE, %edi
        imull $0x9E3779B1, %ecx, %eax
        movl $1024, %edx
1:      movl %eax, (%edi)
        addl $4, %edi
        addl $0x01000193, %eax
        decl %edx
        jnz 1b
next_page:
        incl %ecx
        cmpl $NPAGES, %ecx
        jne fill
        call sum_area
        movl %eax, %ebp
        xorl %ebx, %ebx
        movl %ebx, 0x8000
        movl %ebp, 0x8004
        call report
step:
        incl %ebx
        movl %ebx, 0x8000
        imull $7919, %ebx, %eax
        andl $(NPAGES - 1), %eax
        shll $12, %eax
        movl %ebx, %edx
        shll $2, %edx
        andl $0xffc, %edx
        leal BASE(%eax,%edx), %edi
        imull $0x01000193, %ebx, %edx
        incl %edx
        addl %edx, (%edi)
        addl %edx, %ebp
        movl %ebp, 0x8004
        testl $0x3fff, %ebx
        jnz step
        call report
        jmp step

sum_area:
        xorl %eax, %eax
        movl $BASE, %esi
1:      addl (%esi), %eax
        addl $4, %esi
        cmpl $END, %esi
        jne 1b
        ret

report:
        call sum_area
        movl %eax, %edi
        movl %ebx, %esi
        call hex32
        movb $' ', %al
        outb %al, $0xe9
        movl %edi, %esi
        call hex32
        cmpl %ebp, %edi
        jne bad
        cmpl 0x8004, %ebp
        jne bad
        cmpl 0x8000, %ebx
        jne bad
        movl $0x0a4b4f20, %eax          # " OK\n"
        jmp 2f
bad:    movl $0x44414220, %eax          # " BAD"
        call put4
        movl $0x0a, %eax
        outb %al, $0xe9
        ret
2:      call put4
        ret

put4:   movl $4, %ecx
1:      outb %al, $0xe9
        shrl $8, %eax
        decl %ecx
        jnz 1b
        ret

hex32:  movl $8, %ecx
1:      roll $4, %esi
        movl %esi, %eax
        andl $15, %eax
        cmpl $10, %eax
        jb 2f
        addl $('a' - 10 - '0'), %eax
2:      addl $'0', %eax
        outb %al, $0xe9
        decl %ecx
        jnz 1b
        ret

        .code16
real_start:
        cli
        lgdtl %cs:0xff18
        movl %cr0, %eax
        orl $1, %eax
        movl %eax, %cr0
        ljmpl $0x08, $pm_entry

        .org 0xff00
gdt:    .quad 0
        .quad 0x00cf9a000000ffff
        .quad 0x00cf92000000ffff
        .org 0xff18
        .word 23
        .long 0xffff0000 + 0xff00

        .org 0xfff0
        jmp real_start
        .org 0x10000
