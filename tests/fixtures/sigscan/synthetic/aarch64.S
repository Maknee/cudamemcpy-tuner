/* aarch64.S - synthetic code for the matcher unit tests (llvm-mc, AArch64).
 *
 * Shapes follow GCC's aarch64 libcuda 615 (ldp/madd x/y/z arithmetic, the dst operand addressed through
 * x0 = d + stride, rows copied with memcpy through a PLT stub, the staging buffers initialized through a per-pool
 * helper), with the same non-libcuda offsets as x86_64.S.
 *
 * V selects a variant (regen.cmake assembles each one):
 *   0 base                       engine and buffer init found; internal_memcpy absent (memcpy@plt)
 *   1 duplicate engine           h2h ambiguous
 *   2 bad dst stride             h2h not found
 *   3 in-binary NT copy          the engine calls a local loop with stnp: internal_memcpy found
 *   4 directions 0,1,0,1         buffer init not found (the wrong ones come from a register, so only the full
 *                                symbolic check can reject them, not the constant prefilter)
 *   5 store past the buffer      buffer init not found
 *   6 pool helper                found
 *   7 duplicate staging setup    buffer init ambiguous
 *
 * Layout (must match the Rust scanner tests): engine 0x000, nt_copy 0x400, alloc 0x480, staging 0x500, buffer_init
 * 0x600, pool_init 0x700, memcpy PLT stub 0x800 (GOT slot at image + 0x20010), second engine 0x900, second staging
 * 0xd00, second buffer_init 0xe00; image size 0xf00.
 */
        .text

        .set STRIDE, 0xe0
        .set SOMEPTR1, 0x18
        .set SOMEPTR1_W, 0x28
        .set SOMEPTR1_ADDR, 0x90
        .set SOMEPTR2, 0x48
        .set SOMEPTR2_H, 0x58
        .set SOMEPTR3, 0x70
        .set SOMEPTR3_H, 0x80
        .set UNKX, 0xc0
        .set UNKZ, 0xd0
        .set COPY_W, 0x1d0
        .set COPY_D, 0x1e0
        .set BS, 0x48
        .set POOL0, 0x40
        .set POOL1, 0x100
.if V == 2
        .set BADZ, 8
.else
        .set BADZ, 0
.endif

.macro ENGINE p
\p\()_engine:
        stp x29, x30, [sp, #-0x70]!
        mov x29, sp
        stp x21, x22, [sp, #0x20]
        stp x23, x24, [sp, #0x30]
        stp x25, x26, [sp, #0x40]
        stp x27, x28, [sp, #0x50]
        mov x23, x0
        ldr w0, [x0]
        cmp w0, #2
        b.eq \p\()_src_va
        cmp w0, #3
        b.eq \p\()_src_pg
        cmp w0, #1
        b.eq \p\()_src_obj
        mov x28, #0
        mov x22, #0
        mov x24, #0
\p\()_src_merge:
        ldp x3, x2, [x23, #UNKX]
        add x0, x23, #STRIDE
        ldr x1, [x23, #UNKZ]
        madd x1, x28, x1, x2
        ldr w2, [x23, #STRIDE]
        madd x1, x1, x22, x3
        add x24, x24, x1
        cmp w2, #2
        b.eq \p\()_dst_va
        cmp w2, #3
        b.eq \p\()_dst_pg
        cmp w2, #1
        b.eq \p\()_dst_obj
        mov x27, #0
        mov x21, #0
        mov x25, #0
\p\()_dst_merge:
        ldp x2, x3, [x0, #UNKX]
        ldr x1, [x0, #UNKZ+BADZ]
        madd x0, x27, x1, x3
        ldr x1, [x23, #COPY_D]
        madd x0, x0, x21, x2
        add x25, x25, x0
        cbz x1, \p\()_ret
        mov x26, #0
        stp x19, x20, [sp, #0x10]
\p\()_z:
        ldp x28, x20, [x23, #COPY_W]
        cbz x20, \p\()_ret2
        mov x3, x25
        mov x27, x24
        mov x19, #0
\p\()_y:
        mov x1, x27
        mov x0, x3
        mov x2, x28
        add x19, x19, #1
.if V == 3
        bl nt_copy
.else
        bl memcpy_plt
.endif
        add x27, x27, x22
        add x3, x0, x21
        cmp x20, x19
        b.ne \p\()_y
        add x26, x26, #1
        ldr x0, [x23, #COPY_D]
        cmp x0, x26
        b.hi \p\()_z
\p\()_ret2:
        ldp x19, x20, [sp, #0x10]
\p\()_ret:
        ldp x21, x22, [sp, #0x20]
        ldp x23, x24, [sp, #0x30]
        ldp x25, x26, [sp, #0x40]
        ldp x27, x28, [sp, #0x50]
        ldp x29, x30, [sp], #0x70
        ret
\p\()_src_va:
        ldp x24, x22, [x23, #SOMEPTR3]
        ldr x28, [x23, #SOMEPTR3_H]
        b \p\()_src_merge
\p\()_src_pg:
        ldp x24, x22, [x23, #SOMEPTR2]
        ldr x28, [x23, #SOMEPTR2_H]
        b \p\()_src_merge
\p\()_src_obj:
        ldp x1, x0, [x23, #SOMEPTR1]
        ldp x22, x28, [x23, #SOMEPTR1_W]
        ldr x24, [x1, #SOMEPTR1_ADDR]
        add x24, x24, x0
        b \p\()_src_merge
\p\()_dst_obj:
        ldp x2, x1, [x0, #SOMEPTR1]
        ldp x21, x27, [x0, #SOMEPTR1_W]
        ldr x25, [x2, #SOMEPTR1_ADDR]
        add x25, x25, x1
        b \p\()_dst_merge
\p\()_dst_pg:
        ldp x25, x21, [x0, #SOMEPTR2]
        ldr x27, [x0, #SOMEPTR2_H]
        b \p\()_dst_merge
\p\()_dst_va:
        ldp x25, x21, [x0, #SOMEPTR3]
        ldr x27, [x0, #SOMEPTR3_H]
        b \p\()_dst_merge
.endm

.macro BUFFER_INIT name
\name:
        str xzr, [x1]
        str w2, [x1, #8]
.if V == 5
        str xzr, [x1, #BS]
.else
        str xzr, [x1, #BS-8]
.endif
        mov w0, #0
        ret
.endm

.macro STAGING name, init
\name:
        stp x29, x30, [sp, #-0x30]!
        mov x29, sp
        stp x19, x20, [sp, #0x10]
        mov x19, x0
        mov x0, #1
        mov x1, #0x200
        bl alloc
        mov x20, x0
.if V == 4
        str x21, [sp, #0x20]
        mov w21, #1 /* the wrong directions come from x21: no constant at the call for the prefilter to see */
.endif
.if V == 6
        ldr w3, [x20, #8]
        add x1, x20, #POOL0
        mov x0, x19
        mov w2, #0
        bl pool_init
        ldr w3, [x20, #8]
        add x1, x20, #POOL1
        mov x0, x19
        mov w2, #1
        bl pool_init
.else
        ldr w2, [x20, #8]
        add x1, x20, #POOL0
        mov x0, x19
        mov w3, #0
        bl \init
        ldr w2, [x20, #8]
        add x1, x20, #POOL0+BS
        mov x0, x19
.if V == 4
        mov w3, w21
.else
        mov w3, #0
.endif
        bl \init
        ldr w2, [x20, #8]
        add x1, x20, #POOL1
        mov x0, x19
.if V == 4
        sub w3, w21, #1
.else
        mov w3, #1
.endif
        bl \init
        ldr w2, [x20, #8]
        add x1, x20, #POOL1+BS
        mov x0, x19
        mov w3, #1
        bl \init
.endif
.if V == 4
        ldr x21, [sp, #0x20]
.endif
        ldp x19, x20, [sp, #0x10]
        ldp x29, x30, [sp], #0x30
        ret
.endm

        .org 0x000
        ENGINE e1

        .org 0x400
nt_copy:
        cbz x2, 1f
0:      ldp q0, q1, [x1], #32
        stnp q0, q1, [x0]
        add x0, x0, #32
        subs x2, x2, #32
        b.ne 0b
1:      ret

        .org 0x480
alloc:
        mov x0, #0
        ret

        .org 0x500
        STAGING staging, buffer_init

        .org 0x600
        BUFFER_INIT buffer_init

        .org 0x700
/* (ctx, pool, dir, size) -> buffer_init(ctx, pool, size, dir), buffer_init(ctx, pool + BS, size, dir); the first
   buffer argument is formed by post-index writeback, as GCC does in 615 */
pool_init:
        stp x29, x30, [sp, #-0x40]!
        mov x29, sp
        stp x21, x22, [sp, #0x20]
        mov x21, x1
        mov w22, w3
        stp x19, x20, [sp, #0x10]
        mov x20, x1
        mov w3, w2
        stp x23, x24, [sp, #0x30]
        add x20, x20, #BS+8
        mov w23, w2
        mov x24, x0
        mov w2, w22
        str xzr, [x21], #8
        mov x1, x21
        bl buffer_init
        mov w3, w23
        mov w2, w22
        mov x0, x24
        mov x1, x20
        bl buffer_init
        ldp x19, x20, [sp, #0x10]
        ldp x21, x22, [sp, #0x20]
        ldp x23, x24, [sp, #0x30]
        ldp x29, x30, [sp], #0x40
        ret

        .org 0x800
/* PLT stub: the slot is 0x20010 bytes into the image, i.e. page +0x20000 from this page, offset 0x10 */
memcpy_plt:
        adrp x16, #0x20000
        ldr x17, [x16, #0x10]
        add x16, x16, #0x10
        br x17

        .org 0x900
.if V == 1
        ENGINE e2
.endif

        .org 0xd00
.if V == 7
        STAGING staging2, buffer_init2
.endif

        .org 0xe00
.if V == 7
        BUFFER_INIT buffer_init2
.endif

        .org 0xf00
