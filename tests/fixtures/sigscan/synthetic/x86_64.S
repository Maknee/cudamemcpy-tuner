/* x86_64.S - synthetic code for the matcher unit tests (GNU as, Intel syntax).
 *
 * The shapes follow what GCC emits for libcuda's host-to-host engine, row copy loop and staging-buffer setup, but
 * every offset differs from any libcuda's (stride 0xe0, unkx at 0xc0, width at 0x1d0, pointer1 address 0x90, buffers 0x48
 * apart), so a matcher that passes here reads them from the code rather than knowing them.
 *
 * V selects a variant (regen.cmake assembles each one):
 *   0 base                       everything found
 *   1 duplicate engine           h2h ambiguous (and internal_memcpy, which needs the engine, not found)
 *   2 bad dst stride             the dst unkz is read 8 bytes off: h2h not found
 *   3 no non-temporal store      the engine's callee uses movaps: internal_memcpy not found
 *   4 directions 0,1,0,1         buffer init not found (the wrong ones come from a register, so only the full
 *                                symbolic check can reject them, not the constant prefilter)
 *   5 store past the buffer      buffer init writes 8 bytes beyond the stride: not found
 *   6 pool helper                the four calls go through a per-pool helper: found
 *   7 duplicate staging setup    a second initializer with the same pattern: buffer init ambiguous
 *
 * Layout (must match the Rust scanner tests): engine 0x000, internal_memcpy 0x400, alloc 0x480, staging 0x500, buffer_init
 * 0x600, pool_init 0x700, second engine 0x900, second staging 0xd00, second buffer_init 0xe00; image size 0xf00.
 */
        .intel_syntax noprefix
        .text

        .set STRIDE, 0xe0
        .set SOMEPTR1, 0x18
        .set SOMEPTR1_OFF, 0x20
        .set SOMEPTR1_W, 0x28
        .set SOMEPTR1_H, 0x30
        .set SOMEPTR1_ADDR, 0x90
        .set SOMEPTR2, 0x48
        .set SOMEPTR2_W, 0x50
        .set SOMEPTR2_H, 0x58
        .set SOMEPTR3, 0x70
        .set SOMEPTR3_W, 0x78
        .set SOMEPTR3_H, 0x80
        .set UNKX, 0xc0
        .set UNKY, 0xc8
        .set UNKZ, 0xd0
        .set COPY_W, 0x1d0
        .set COPY_H, 0x1d8
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
        push rbp
        mov rbp, rsp
        push r15
        push r14
        push r13
        push r12
        push rbx
        sub rsp, 0x48
        mov eax, dword ptr [rdi]
        mov qword ptr [rbp-0x50], rdi
        cmp eax, 2
        je \p\()_src_va
        cmp eax, 3
        je \p\()_src_pg
        cmp eax, 1
        je \p\()_src_obj
        mov qword ptr [rbp-0x38], 0
        xor edx, edx
        xor ecx, ecx
\p\()_src_merge:
        mov rsi, qword ptr [rbp-0x50]
        mov rax, qword ptr [rsi+UNKZ]
        imul rax, rdx
        add rax, qword ptr [rsi+UNKY]
        imul rax, qword ptr [rbp-0x38]
        add rax, qword ptr [rsi+UNKX]
        add rax, rcx
        mov qword ptr [rbp-0x58], rax
        mov eax, dword ptr [rsi+STRIDE]
        cmp eax, 2
        je \p\()_dst_va
        cmp eax, 3
        je \p\()_dst_pg
        cmp eax, 1
        je \p\()_dst_obj
        mov qword ptr [rbp-0x40], 0
        xor ecx, ecx
        xor esi, esi
\p\()_dst_merge:
        mov rdi, qword ptr [rbp-0x50]
        mov rbx, qword ptr [rbp-0x40]
        mov rax, qword ptr [rdi+STRIDE+UNKZ+BADZ]
        imul rax, rcx
        add rax, qword ptr [rdi+STRIDE+UNKY]
        imul rax, rbx
        add rax, qword ptr [rdi+STRIDE+UNKX]
        add rax, rsi
        cmp qword ptr [rdi+COPY_D], 0
        mov qword ptr [rbp-0x60], rax
        je \p\()_ret
        mov qword ptr [rbp-0x48], 0
\p\()_z:
        mov rax, qword ptr [rbp-0x50]
        mov rbx, qword ptr [rax+COPY_H]
        mov r12, qword ptr [rax+COPY_W]
        test rbx, rbx
        je \p\()_ret
        mov r14, qword ptr [rbp-0x60]
        mov r13, qword ptr [rbp-0x58]
        xor r15d, r15d
\p\()_y:
        mov rsi, r13
        mov rdi, r14
        mov rdx, r12
        add r15, 1
        call internal_memcpy
        add r13, qword ptr [rbp-0x38]
        add r14, qword ptr [rbp-0x40]
        cmp rbx, r15
        jne \p\()_y
        mov rcx, qword ptr [rbp-0x50]
        add qword ptr [rbp-0x48], 1
        mov rax, qword ptr [rbp-0x48]
        cmp qword ptr [rcx+COPY_D], rax
        ja \p\()_z
\p\()_ret:
        add rsp, 0x48
        pop rbx
        pop r12
        pop r13
        pop r14
        pop r15
        pop rbp
        ret
\p\()_src_va:
        mov rax, qword ptr [rbp-0x50]
        mov rdx, qword ptr [rax+SOMEPTR3_W]
        mov rcx, qword ptr [rax+SOMEPTR3]
        mov qword ptr [rbp-0x38], rdx
        mov rdx, qword ptr [rax+SOMEPTR3_H]
        jmp \p\()_src_merge
\p\()_src_pg:
        mov rax, qword ptr [rbp-0x50]
        mov rdx, qword ptr [rax+SOMEPTR2_W]
        mov rcx, qword ptr [rax+SOMEPTR2]
        mov qword ptr [rbp-0x38], rdx
        mov rdx, qword ptr [rax+SOMEPTR2_H]
        jmp \p\()_src_merge
\p\()_src_obj:
        mov rdx, qword ptr [rdi+SOMEPTR1_W]
        mov rax, qword ptr [rdi+SOMEPTR1]
        mov rcx, qword ptr [rdi+SOMEPTR1_OFF]
        mov qword ptr [rbp-0x38], rdx
        add rcx, qword ptr [rax+SOMEPTR1_ADDR]
        mov rdx, qword ptr [rdi+SOMEPTR1_H]
        jmp \p\()_src_merge
\p\()_dst_obj:
        mov rax, qword ptr [rsi+STRIDE+SOMEPTR1]
        mov rcx, rsi
        mov rsi, qword ptr [rsi+STRIDE+SOMEPTR1_OFF]
        add rsi, qword ptr [rax+SOMEPTR1_ADDR]
        mov rax, rcx
        mov rcx, qword ptr [rcx+STRIDE+SOMEPTR1_W]
        mov qword ptr [rbp-0x40], rcx
        mov rcx, qword ptr [rax+STRIDE+SOMEPTR1_H]
        jmp \p\()_dst_merge
\p\()_dst_pg:
        mov rax, rsi
        mov rsi, qword ptr [rsi+STRIDE+SOMEPTR2]
        mov rcx, qword ptr [rax+STRIDE+SOMEPTR2_W]
        mov qword ptr [rbp-0x40], rcx
        mov rcx, qword ptr [rax+STRIDE+SOMEPTR2_H]
        jmp \p\()_dst_merge
\p\()_dst_va:
        mov rax, rsi
        mov rsi, qword ptr [rsi+STRIDE+SOMEPTR3]
        mov rcx, qword ptr [rax+STRIDE+SOMEPTR3_W]
        mov qword ptr [rbp-0x40], rcx
        mov rcx, qword ptr [rax+STRIDE+SOMEPTR3_H]
        jmp \p\()_dst_merge
.endm

/* (ctx, buf, size, dir): zeroes the BS-byte buffer header, records the size */
.macro BUFFER_INIT name
\name:
        push rbp
        mov rbp, rsp
        mov qword ptr [rsi], 0
        mov dword ptr [rsi+8], edx
.if V == 5
        mov qword ptr [rsi+BS], 0
.else
        mov qword ptr [rsi+BS-8], 0
.endif
        xor eax, eax
        pop rbp
        ret
.endm

/* the staging object: calloc'ed, two pools of two buffers, each initialized with (ctx, buf, size, dir) */
.macro STAGING name, init
\name:
        push rbp
        mov rbp, rsp
        push r14
        push r12
        push rbx
        sub rsp, 8
        mov r12, rdi
        mov edi, 1
        mov esi, 0x200
        call alloc
        mov r14, rax
.if V == 4
        mov ebx, 1 /* the wrong directions come from rbx: no constant at the call for the prefilter to see */
.endif
.if V == 6
        mov ecx, dword ptr [r14+8]
        lea rsi, [r14+POOL0]
        mov rdi, r12
        xor edx, edx
        call pool_init
        mov ecx, dword ptr [r14+8]
        lea rsi, [r14+POOL1]
        mov rdi, r12
        mov edx, 1
        call pool_init
.else
        mov edx, dword ptr [r14+8]
        lea rsi, [r14+POOL0]
        mov rdi, r12
        xor ecx, ecx
        call \init
        mov edx, dword ptr [r14+8]
        lea rsi, [r14+POOL0+BS]
        mov rdi, r12
.if V == 4
        mov ecx, ebx
.else
        xor ecx, ecx
.endif
        call \init
        mov edx, dword ptr [r14+8]
        lea rsi, [r14+POOL1]
        mov rdi, r12
.if V == 4
        lea ecx, [rbx-1]
.else
        mov ecx, 1
.endif
        call \init
        mov edx, dword ptr [r14+8]
        lea rsi, [r14+POOL1+BS]
        mov rdi, r12
        mov ecx, 1
        call \init
.endif
        add rsp, 8
        pop rbx
        pop r12
        pop r14
        pop rbp
        ret
.endm

        .org 0x000
        ENGINE e1

        .org 0x400
internal_memcpy:
        push rbp
        mov rbp, rsp
qc_loop:
        test rdx, rdx
        je qc_done
        movaps xmm0, xmmword ptr [rsi]
.if V == 3
        movaps xmmword ptr [rdi], xmm0
.else
        movntps xmmword ptr [rdi], xmm0
.endif
        add rsi, 16
        add rdi, 16
        sub rdx, 16
        jmp qc_loop
qc_done:
        pop rbp
        ret

        .org 0x480
alloc:
        xor eax, eax
        ret

        .org 0x500
        STAGING staging, buffer_init

        .org 0x600
        BUFFER_INIT buffer_init

        .org 0x700
/* (ctx, pool, dir, size) -> buffer_init(ctx, pool, size, dir), buffer_init(ctx, pool + BS, size, dir) */
pool_init:
        push rbp
        mov rbp, rsp
        push r15
        push r14
        push r13
        push r12
        mov r12, rdi
        mov r13, rsi
        mov r14d, edx
        mov r15d, ecx
        mov rdi, r12
        mov rsi, r13
        mov edx, r15d
        mov ecx, r14d
        call buffer_init
        mov rdi, r12
        lea rsi, [r13+BS]
        mov edx, r15d
        mov ecx, r14d
        call buffer_init
        pop r12
        pop r13
        pop r14
        pop r15
        pop rbp
        ret

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
