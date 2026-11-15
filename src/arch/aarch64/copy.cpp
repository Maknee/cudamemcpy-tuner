#include "arch/copy.h"

#if !defined(__aarch64__)
#error "arch/aarch64/copy.cpp is the AArch64 implementation; CMake picks one per architecture (CMT_ARCH)"
#endif

namespace cmt {
namespace hwcopy {

namespace {

constexpr std::size_t kStep = 64;

inline void CopyBytes(uint8_t* dst, const uint8_t* src, std::size_t n)
{
	for (std::size_t i = 0; i < n; i++)
	{
		dst[i] = src[i];
	}
}

inline std::size_t HeadBytes(const void* p, std::size_t align, std::size_t bytes)
{
	const std::size_t mis = reinterpret_cast<uintptr_t>(p) & (align - 1);
	const std::size_t head = mis == 0 ? 0 : align - mis;
	return head < bytes ? head : bytes;
}

enum class Hint : uint8_t
{
	Cached,
	NtStore,
	NtLoad
};

template <Hint H>
inline void NeonBody(void* destination, const void* source, std::size_t bytes)
{
	auto* dst = static_cast<uint8_t*>(destination);
	auto* src = static_cast<const uint8_t*>(source);
	const std::size_t head = HeadBytes(H == Hint::NtLoad ? static_cast<const void*>(src) : dst, 16, bytes);
	CopyBytes(dst, src, head);
	dst += head;
	src += head;
	bytes -= head;
	const std::size_t body = bytes & ~(kStep - 1);
	std::size_t i = 0;
	for (; i < body; i += kStep)
	{
		if constexpr (H == Hint::Cached)
		{
			__asm__ __volatile__("ldp q0, q1, [%[s]]\n\t"
								 "ldp q2, q3, [%[s], #32]\n\t"
								 "stp q0, q1, [%[d]]\n\t"
								 "stp q2, q3, [%[d], #32]"
								 :
								 : [s] "r"(src + i), [d] "r"(dst + i)
								 : "v0", "v1", "v2", "v3", "memory");
		}
		else if constexpr (H == Hint::NtStore)
		{
			__asm__ __volatile__("ldp q0, q1, [%[s]]\n\t"
								 "ldp q2, q3, [%[s], #32]\n\t"
								 "stnp q0, q1, [%[d]]\n\t"
								 "stnp q2, q3, [%[d], #32]"
								 :
								 : [s] "r"(src + i), [d] "r"(dst + i)
								 : "v0", "v1", "v2", "v3", "memory");
		}
		else
		{
			__asm__ __volatile__("ldnp q0, q1, [%[s]]\n\t"
								 "ldnp q2, q3, [%[s], #32]\n\t"
								 "stp q0, q1, [%[d]]\n\t"
								 "stp q2, q3, [%[d], #32]"
								 :
								 : [s] "r"(src + i), [d] "r"(dst + i)
								 : "v0", "v1", "v2", "v3", "memory");
		}
	}
	CopyBytes(dst + i, src + i, bytes - i);
}

std::size_t ReadDcacheLine()
{
	uint64_t ctr = 0;
	__asm__ __volatile__("mrs %0, ctr_el0" : "=r"(ctr));
	return std::size_t{4} << ((ctr >> 16) & 0xf);
}

template <WriteBack W>
void MaintainRange(const void* ptr, std::size_t bytes)
{
	if (bytes == 0)
	{
		return;
	}
	const std::size_t line = WriteBackLineBytes();
	uintptr_t p = reinterpret_cast<uintptr_t>(ptr) & ~uintptr_t(line - 1);
	const uintptr_t end = reinterpret_cast<uintptr_t>(ptr) + bytes;
	for (; p < end; p += line)
	{
		if constexpr (W == WriteBack::Flush)
		{
			__asm__ __volatile__("dc civac, %0" : : "r"(p) : "memory");
		}
		else
		{
			__asm__ __volatile__("dc cvac, %0" : : "r"(p) : "memory");
		}
	}
}

} // namespace

__attribute__((noinline)) void CopyCachedNeon(void* dst, const void* src, std::size_t bytes)
{
	NeonBody<Hint::Cached>(dst, src, bytes);
}

__attribute__((noinline)) void CopyNtStoreNeon(void* dst, const void* src, std::size_t bytes)
{
	NeonBody<Hint::NtStore>(dst, src, bytes);
}

__attribute__((noinline)) void CopyNtLoadNeon(void* dst, const void* src, std::size_t bytes)
{
	NeonBody<Hint::NtLoad>(dst, src, bytes);
}

#define CMT_SVE_COPY_KERNEL(fn, load, store)                                                                 \
	__attribute__((target("+sve"), noinline)) void fn(void* dst, const void* src, std::size_t bytes)         \
	{                                                                                                        \
		std::size_t i = 0;                                                                                   \
		__asm__ __volatile__("whilelo p0.b, %[i], %[n]\n\t"                                                  \
							 "b.none 2f\n\t"                                                                 \
							 "1:\n\t" load " {z0.b}, p0/z, [%[s], %[i]]\n\t" store                           \
							 " {z0.b}, p0, [%[d], %[i]]\n\t"                                                 \
							 "incb %[i]\n\t"                                                                 \
							 "whilelo p0.b, %[i], %[n]\n\t"                                                  \
							 "b.first 1b\n\t"                                                                \
							 "2:"                                                                            \
							 : [i] "+r"(i)                                                                   \
							 : [s] "r"(src), [d] "r"(dst), [n] "r"(bytes)                                    \
							 : "z0", "p0", "memory", "cc");                                                  \
	}

CMT_SVE_COPY_KERNEL(CopyCachedSve, "ld1b", "st1b")
CMT_SVE_COPY_KERNEL(CopyNtStoreSve, "ld1b", "stnt1b")
CMT_SVE_COPY_KERNEL(CopyNtLoadSve, "ldnt1b", "st1b")

#undef CMT_SVE_COPY_KERNEL

std::size_t WriteBackLineBytes()
{
	static const std::size_t line = ReadDcacheLine();
	return line;
}

void WriteBackLines(WriteBack wb, const void* p, std::size_t bytes)
{
	if (wb == WriteBack::Clean)
	{
		MaintainRange<WriteBack::Clean>(p, bytes);
	}
	else if (wb == WriteBack::Flush)
	{
		MaintainRange<WriteBack::Flush>(p, bytes);
	}
}

// DSB waits for cache maintenance; SY includes DMA outside the inner-shareable domain.
void WriteBackFence()
{
	__asm__ __volatile__("dsb sy" ::: "memory");
}

} // namespace hwcopy

// DMA needs outer-shareable ordering. Not yet verified on AArch64 hardware.
void StoreFence()
{
	__asm__ __volatile__("dmb osh" ::: "memory");
}

} // namespace cmt
