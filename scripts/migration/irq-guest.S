# SPDX-License-Identifier: GPL-2.0-or-later
#
# Interrupt-driven migration test guest: a 64 KiB -bios image for -M q35.
#
# It enters flat protected mode, sets up an IDT, the 8259 pair (vectors 0x20 and 0x28), the
# PIT at 100 Hz on IRQ0, COM1 with its FIFO on IRQ4, and the local APIC: LINT0 as ExtINT so the
# 8259 reaches the CPU, and the APIC timer periodic at 100 Hz on vector 0x40. Then it fills
# 1 MiB at 1 MiB (every fourth page left zero) and loops with interrupts on, adding a value
# derived from the step number to one dword per step and keeping a running sum in %ebp.
#
# Every 25 PIT ticks the loop sums the area and queues a line for COM1:
#
#     REPORT PIT_TICKS APIC_TICKS STEP SUM OK
#
# The serial port is only ever written by its THRE interrupt handler, which drains a ring
# buffer in RAM, so output stops if any of the 8259, the PIT, the APIC or the UART loses its
# state across a migration. An exception prints "EXC" and halts.
#
# Build: as --32 -o irq-guest.o irq-guest.S
#        ld -m elf_i386 -Ttext=0xffff0000 -e 0xffff0000 -o irq-guest.elf irq-guest.o
#        objcopy -O binary irq-guest.elf irq-guest.bin
        .set BASE, 0x100000
        .set NPAGES, 256
        .set END, BASE + NPAGES * 4096
        .set IDT, 0x9000
        .set PIT_TICKS, 0x9800
        .set APIC_TICKS, 0x9804
        .set REPORT, 0x9808
        .set NEXT_TICK, 0x980c
        .set HEAD, 0x9810
        .set TAIL, 0x9814
        .set RING, 0xa000
        .set RING_MASK, 0xfff
        .set LAPIC, 0xfee00000
        .set COM1, 0x3f8
        .set TICKS_PER_REPORT, 25
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
        xorl %eax, %eax
        movl $PIT_TICKS, %edi
        movl $8, %ecx
        rep stosl

        # The IDT: exceptions, the 8259 vectors, the APIC timer, everything else spurious.
        xorl %ecx, %ecx
idt_loop:
        movl $spurious, %eax
        cmpl $32, %ecx
        jae 1f
        movl $exception, %eax
        jmp 3f
1:      cmpl $40, %ecx
        jae 2f
        movl $irq_master, %eax
        jmp 3f
2:      cmpl $48, %ecx
        jae 3f
        movl $irq_slave, %eax
3:      call set_gate
        incl %ecx
        cmpl $256, %ecx
        jne idt_loop
        movl $0x20, %ecx
        movl $pit_irq, %eax
        call set_gate
        movl $0x24, %ecx
        movl $serial_irq, %eax
        call set_gate
        movl $0x40, %ecx
        movl $apic_timer_irq, %eax
        call set_gate
        lidtl idt_desc

        # The 8259 pair: edge triggered, cascaded on IRQ2, IRQ0 and IRQ4 unmasked.
        movb $0x11, %al
        outb %al, $0x20
        outb %al, $0xa0
        movb $0x20, %al
        outb %al, $0x21
        movb $0x28, %al
        outb %al, $0xa1
        movb $0x04, %al
        outb %al, $0x21
        movb $0x02, %al
        outb %al, $0xa1
        movb $0x01, %al
        outb %al, $0x21
        outb %al, $0xa1
        movb $0xea, %al
        outb %al, $0x21
        movb $0xff, %al
        outb %al, $0xa1

        # PIT channel 0: rate generator at 1193182 / 11932 Hz.
        movb $0x34, %al
        outb %al, $0x43
        movw $11932, %ax
        outb %al, $0x40
        movb %ah, %al
        outb %al, $0x40

        # COM1: 115200 8N1, FIFO on, OUT2, no interrupts until there is something to send.
        movw $COM1 + 3, %dx
        movb $0x80, %al
        outb %al, %dx
        movw $COM1, %dx
        movb $1, %al
        outb %al, %dx
        movw $COM1 + 1, %dx
        xorb %al, %al
        outb %al, %dx
        movw $COM1 + 3, %dx
        movb $0x03, %al
        outb %al, %dx
        movw $COM1 + 2, %dx
        movb $0x07, %al
        outb %al, %dx
        movw $COM1 + 4, %dx
        movb $0x0b, %al
        outb %al, %dx
        movw $COM1 + 1, %dx
        xorb %al, %al
        outb %al, %dx

        # The local APIC: enabled, LINT0 ExtINT, timer periodic every 10 ms.
        movl $0x1ff, LAPIC + 0xf0
        movl $0x700, LAPIC + 0x350
        movl $0x0b, LAPIC + 0x3e0
        movl $0x20040, LAPIC + 0x320
        movl $10000000, LAPIC + 0x380

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
        sti
step:
        incl %ebx
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
        movl PIT_TICKS, %eax
        cmpl NEXT_TICK, %eax
        jb step
        call report
        addl $TICKS_PER_REPORT, NEXT_TICK
        jmp step

# set_gate: IDT entry %ecx points at %eax, a 32-bit interrupt gate. Clobbers %edx, %edi.
set_gate:
        leal IDT(,%ecx,8), %edi
        movl %eax, %edx
        andl $0xffff, %edx
        orl $0x00080000, %edx
        movl %edx, (%edi)
        movl %eax, %edx
        andl $0xffff0000, %edx
        orl $0x8e00, %edx
        movl %edx, 4(%edi)
        ret

sum_area:
        xorl %eax, %eax
        movl $BASE, %esi
1:      addl (%esi), %eax
        addl $4, %esi
        cmpl $END, %esi
        jne 1b
        ret

# report: queues "REPORT PIT APIC STEP SUM OK" (or BAD) for the step in %ebx.
report:
        call sum_area
        movl %eax, %edi
        movl REPORT, %esi
        call hex32
        call space
        movl PIT_TICKS, %esi
        call hex32
        call space
        movl APIC_TICKS, %esi
        call hex32
        call space
        movl %ebx, %esi
        call hex32
        call space
        movl %edi, %esi
        call hex32
        incl REPORT
        cmpl %ebp, %edi
        jne 1f
        movb $' ', %al
        call putc
        movb $'O', %al
        call putc
        movb $'K', %al
        call putc
        jmp 2f
1:      movb $' ', %al
        call putc
        movb $'B', %al
        call putc
        movb $'A', %al
        call putc
        movb $'D', %al
        call putc
2:      movb $0x0a, %al
        jmp putc

space:  movb $' ', %al
        jmp putc

hex32:  movl $8, %ecx
1:      roll $4, %esi
        movl %esi, %eax
        andl $15, %eax
        cmpl $10, %eax
        jb 2f
        addl $('a' - 10 - '0'), %eax
2:      addl $'0', %eax
        call putc
        decl %ecx
        jnz 1b
        ret

# putc: queues %al for COM1 and turns the THRE interrupt on. Waits with interrupts on while
# the ring is full.
putc:   pushl %esi
        pushl %edx
        pushl %eax
1:      cli
        movl HEAD, %esi
        leal 1(%esi), %edx
        andl $RING_MASK, %edx
        cmpl TAIL, %edx
        jne 2f
        sti
        pause
        jmp 1b
2:      movb (%esp), %al
        movb %al, RING(%esi)
        movl %edx, HEAD
        movw $COM1 + 1, %dx
        movb $0x02, %al
        outb %al, %dx
        sti
        popl %eax
        popl %edx
        popl %esi
        ret

pit_irq:
        pushl %eax
        incl PIT_TICKS
        movb $0x20, %al
        outb %al, $0x20
        popl %eax
        iret

apic_timer_irq:
        incl APIC_TICKS
        movl $0, LAPIC + 0xb0
        iret

# serial_irq: services COM1 until IIR says nothing is pending. THRE moves up to 16 bytes
# from the ring into the FIFO, or turns THRE off when the ring is empty.
serial_irq:
        pushal
1:      movw $COM1 + 2, %dx
        inb %dx, %al
        testb $1, %al
        jnz 9f
        andb $0x0e, %al
        cmpb $0x02, %al
        jne 5f
        movl $16, %ecx
2:      movl TAIL, %esi
        cmpl HEAD, %esi
        je 4f
        movb RING(%esi), %al
        movw $COM1, %dx
        outb %al, %dx
        incl %esi
        andl $RING_MASK, %esi
        movl %esi, TAIL
        decl %ecx
        jnz 2b
        jmp 1b
4:      movw $COM1 + 1, %dx
        xorb %al, %al
        outb %al, %dx
        jmp 1b
5:      cmpb $0x06, %al
        jne 6f
        movw $COM1 + 5, %dx
        inb %dx, %al
        jmp 1b
6:      testb %al, %al
        jne 7f
        movw $COM1 + 6, %dx
        inb %dx, %al
        jmp 1b
7:      movw $COM1, %dx
        inb %dx, %al
        jmp 1b
9:      movb $0x20, %al
        outb %al, $0x20
        popal
        iret

irq_master:
        pushl %eax
        movb $0x20, %al
        outb %al, $0x20
        popl %eax
        iret

irq_slave:
        pushl %eax
        movb $0x20, %al
        outb %al, $0xa0
        outb %al, $0x20
        popl %eax
        iret

spurious:
        iret

# exception: prints "EXC" to COM1 without interrupts and to the debugcon port, then stops.
exception:
        cli
        movl $0x0a435845, %ebx
        movl $4, %ecx
1:      movw $COM1 + 5, %dx
2:      inb %dx, %al
        testb $0x20, %al
        jz 2b
        movb %bl, %al
        movw $COM1, %dx
        outb %al, %dx
        outb %al, $0xe9
        shrl $8, %ebx
        decl %ecx
        jnz 1b
3:      hlt
        jmp 3b

idt_desc:
        .word 256 * 8 - 1
        .long IDT

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
