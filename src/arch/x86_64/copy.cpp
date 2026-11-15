#include "arch/copy.h"

// Per-function ISA targets keep shared inline helpers safe on baseline CPUs.
// AVX2/AVX-512 loops and SSE2 nt_store are ported from copies.c.

#if !defined(__x86_64__)
#error "arch/x86_64/copy.cpp is the x86-64 implementation; CMake picks one per architecture (CMT_ARCH)"
#endif

namespace cmt {
namespace hwcopy {

namespace {

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

template <class T>
inline T* NtLoadPtr(const uint8_t* p)
{
	return reinterpret_cast<T*>(const_cast<uint8_t*>(p));
}

constexpr std::size_t kPrefetchDistance = 1024;
constexpr std::size_t kPrefetchBlock = 128;

struct Sse2
{
	static constexpr std::size_t kBytes = 16;

	static void Cached(uint8_t* d, const uint8_t* s)
	{
		_mm_store_ps(reinterpret_cast<float*>(d), _mm_loadu_ps(reinterpret_cast<const float*>(s)));
	}

	static void NtStore(uint8_t* d, const uint8_t* s)
	{
		_mm_stream_ps(reinterpret_cast<float*>(d), _mm_loadu_ps(reinterpret_cast<const float*>(s)));
	}

	static void NtStoreBlock(uint8_t* d, const uint8_t* s)
	{
		__m128 v[kPrefetchBlock / kBytes];
		for (std::size_t k = 0; k < kPrefetchBlock / kBytes; k++)
		{
			v[k] = _mm_loadu_ps(reinterpret_cast<const float*>(s + k * kBytes));
		}
		for (std::size_t k = 0; k < kPrefetchBlock / kBytes; k++)
		{
			_mm_stream_ps(reinterpret_cast<float*>(d + k * kBytes), v[k]);
		}
	}
};

struct Avx2
{
	static constexpr std::size_t kBytes = 32;

	__attribute__((target("avx2"))) static void Cached(uint8_t* d, const uint8_t* s)
	{
		_mm256_store_si256(reinterpret_cast<__m256i*>(d),
						   _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s)));
	}

	__attribute__((target("avx2"))) static void NtStore(uint8_t* d, const uint8_t* s)
	{
		_mm256_stream_si256(reinterpret_cast<__m256i*>(d),
							_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s)));
	}

	__attribute__((target("avx2"))) static void NtLoad(uint8_t* d, const uint8_t* s)
	{
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(d), _mm256_stream_load_si256(NtLoadPtr<__m256i>(s)));
	}

	__attribute__((target("avx2"))) static void NtStoreBlock(uint8_t* d, const uint8_t* s)
	{
		const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
		const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + 32));
		const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + 64));
		const __m256i e = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + 96));
		_mm256_stream_si256(reinterpret_cast<__m256i*>(d), a);
		_mm256_stream_si256(reinterpret_cast<__m256i*>(d + 32), b);
		_mm256_stream_si256(reinterpret_cast<__m256i*>(d + 64), c);
		_mm256_stream_si256(reinterpret_cast<__m256i*>(d + 96), e);
	}
};

struct Avx512
{
	static constexpr std::size_t kBytes = 64;

	__attribute__((target("avx512f"))) static void Cached(uint8_t* d, const uint8_t* s)
	{
		_mm512_store_si512(d, _mm512_loadu_si512(s));
	}

	__attribute__((target("avx512f"))) static void NtStore(uint8_t* d, const uint8_t* s)
	{
		_mm512_stream_si512(reinterpret_cast<__m512i*>(d), _mm512_loadu_si512(s));
	}

	__attribute__((target("avx512f"))) static void NtLoad(uint8_t* d, const uint8_t* s)
	{
		_mm512_storeu_si512(d, _mm512_stream_load_si512(NtLoadPtr<void>(s)));
	}

	__attribute__((target("avx512f"))) static void NtStoreBlock(uint8_t* d, const uint8_t* s)
	{
		const __m512i a = _mm512_loadu_si512(s);
		const __m512i b = _mm512_loadu_si512(s + 64);
		_mm512_stream_si512(reinterpret_cast<__m512i*>(d), a);
		_mm512_stream_si512(reinterpret_cast<__m512i*>(d + 64), b);
	}
};

template <class V>
inline void CachedBody(void* destination, const void* source, std::size_t bytes)
{
	auto* dst = static_cast<uint8_t*>(destination);
	auto* src = static_cast<const uint8_t*>(source);
	const std::size_t head = HeadBytes(dst, V::kBytes, bytes);
	CopyBytes(dst, src, head);
	dst += head;
	src += head;
	bytes -= head;
	const std::size_t vec_bytes = bytes & ~(V::kBytes - 1);
	std::size_t i = 0;
	for (; i < vec_bytes; i += V::kBytes)
	{
		V::Cached(dst + i, src + i);
	}
	CopyBytes(dst + i, src + i, bytes - i);
}

template <class V>
inline void NtStoreBody(void* destination, const void* source, std::size_t bytes)
{
	auto* dst = static_cast<uint8_t*>(destination);
	auto* src = static_cast<const uint8_t*>(source);
	const std::size_t head = HeadBytes(dst, V::kBytes, bytes);
	CopyBytes(dst, src, head);
	dst += head;
	src += head;
	bytes -= head;
	const std::size_t vec_bytes = bytes & ~(V::kBytes - 1);
	std::size_t i = 0;
	for (; i < vec_bytes; i += V::kBytes)
	{
		V::NtStore(dst + i, src + i);
	}
	CopyBytes(dst + i, src + i, bytes - i);
}

template <class V>
inline void NtLoadBody(void* destination, const void* source, std::size_t bytes)
{
	auto* dst = static_cast<uint8_t*>(destination);
	auto* src = static_cast<const uint8_t*>(source);
	const std::size_t head = HeadBytes(src, V::kBytes, bytes);
	CopyBytes(dst, src, head);
	dst += head;
	src += head;
	bytes -= head;
	const std::size_t vec_bytes = bytes & ~(V::kBytes - 1);
	std::size_t i = 0;
	for (; i < vec_bytes; i += V::kBytes)
	{
		V::NtLoad(dst + i, src + i);
	}
	CopyBytes(dst + i, src + i, bytes - i);
}

template <class V>
inline void NtStorePrefetchBody(void* destination, const void* source, std::size_t bytes)
{
	auto* dst = static_cast<uint8_t*>(destination);
	auto* src = static_cast<const uint8_t*>(source);
	const std::size_t head = HeadBytes(dst, V::kBytes, bytes);
	CopyBytes(dst, src, head);
	dst += head;
	src += head;
	bytes -= head;
	const std::size_t block_bytes = bytes & ~(kPrefetchBlock - 1);
	std::size_t i = 0;
	for (; i < block_bytes; i += kPrefetchBlock)
	{
		_mm_prefetch(reinterpret_cast<const char*>(src + i + kPrefetchDistance), _MM_HINT_T0);
		_mm_prefetch(reinterpret_cast<const char*>(src + i + kPrefetchDistance + 64), _MM_HINT_T0);
		V::NtStoreBlock(dst + i, src + i);
	}
	const std::size_t vec_bytes = bytes & ~(V::kBytes - 1);
	for (; i < vec_bytes; i += V::kBytes)
	{
		V::NtStore(dst + i, src + i);
	}
	CopyBytes(dst + i, src + i, bytes - i);
}

} // namespace

__attribute__((noinline)) void CopyMovsb(void* dst, const void* src, std::size_t bytes)
{
	__asm__ __volatile__("rep movsb" : "+D"(dst), "+S"(src), "+c"(bytes) : : "memory");
}

#define CMT_X86_COPY_KERNEL(fn, body, V, target_attr)                                                        \
	__attribute__((target_attr noinline, flatten)) void fn(void* d, const void* s, std::size_t n)            \
	{                                                                                                        \
		body<V>(d, s, n);                                                                                    \
	}
#define CMT_SSE2_TARGET
#define CMT_AVX2_TARGET target("avx2"),
#define CMT_AVX512_TARGET target("avx512f"),

CMT_X86_COPY_KERNEL(CopyCachedSse2, CachedBody, Sse2, CMT_SSE2_TARGET)
CMT_X86_COPY_KERNEL(CopyCachedAvx2, CachedBody, Avx2, CMT_AVX2_TARGET)
CMT_X86_COPY_KERNEL(CopyCachedAvx512, CachedBody, Avx512, CMT_AVX512_TARGET)
CMT_X86_COPY_KERNEL(CopyNtStoreSse2, NtStoreBody, Sse2, CMT_SSE2_TARGET)
CMT_X86_COPY_KERNEL(CopyNtStoreAvx2, NtStoreBody, Avx2, CMT_AVX2_TARGET)
CMT_X86_COPY_KERNEL(CopyNtStoreAvx512, NtStoreBody, Avx512, CMT_AVX512_TARGET)
CMT_X86_COPY_KERNEL(CopyNtLoadAvx2, NtLoadBody, Avx2, CMT_AVX2_TARGET)
CMT_X86_COPY_KERNEL(CopyNtLoadAvx512, NtLoadBody, Avx512, CMT_AVX512_TARGET)
CMT_X86_COPY_KERNEL(CopyNtStorePrefetchSse2, NtStorePrefetchBody, Sse2, CMT_SSE2_TARGET)
CMT_X86_COPY_KERNEL(CopyNtStorePrefetchAvx2, NtStorePrefetchBody, Avx2, CMT_AVX2_TARGET)
CMT_X86_COPY_KERNEL(CopyNtStorePrefetchAvx512, NtStorePrefetchBody, Avx512, CMT_AVX512_TARGET)

#undef CMT_X86_COPY_KERNEL
#undef CMT_SSE2_TARGET
#undef CMT_AVX2_TARGET
#undef CMT_AVX512_TARGET

namespace {

bool CpuHasLeaf7EbxBit(unsigned bit)
{
	unsigned a = 0;
	unsigned b = 0;
	unsigned c = 0;
	unsigned d = 0;
	return __get_cpuid_count(7, 0, &a, &b, &c, &d) != 0 && (b & (1u << bit)) != 0;
}

bool CpuHasClflushopt()
{
	static const bool v = CpuHasLeaf7EbxBit(23);
	return v;
}

bool CpuHasClwb()
{
	static const bool v = CpuHasLeaf7EbxBit(24);
	return v;
}

__attribute__((target("clflushopt"))) void FlushRangeOpt(const uint8_t* p, const uint8_t* end)
{
	for (; p < end; p += 64)
	{
		_mm_clflushopt(const_cast<uint8_t*>(p));
	}
}

__attribute__((target("clwb"))) void ClwbRangeOpt(const uint8_t* p, const uint8_t* end)
{
	for (; p < end; p += 64)
	{
		_mm_clwb(const_cast<uint8_t*>(p));
	}
}

void FlushRange(const void* ptr, std::size_t bytes)
{
	if (bytes == 0)
	{
		return;
	}
	const uint8_t* p = reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(ptr) & ~uintptr_t(63));
	const uint8_t* end = static_cast<const uint8_t*>(ptr) + bytes;
	if (CpuHasClflushopt())
	{
		FlushRangeOpt(p, end);
	}
	else
	{
		for (; p < end; p += 64)
		{
			_mm_clflush(p);
		}
	}
}

void ClwbRange(const void* ptr, std::size_t bytes)
{
	if (bytes == 0)
	{
		return;
	}
	if (!CpuHasClwb())
	{
		FlushRange(ptr, bytes);
		return;
	}
	const uint8_t* p = reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(ptr) & ~uintptr_t(63));
	ClwbRangeOpt(p, static_cast<const uint8_t*>(ptr) + bytes);
}

} // namespace

std::size_t WriteBackLineBytes()
{
	return 64;
}

void WriteBackLines(WriteBack wb, const void* p, std::size_t bytes)
{
	if (wb == WriteBack::Clean)
	{
		ClwbRange(p, bytes);
	}
	else if (wb == WriteBack::Flush)
	{
		FlushRange(p, bytes);
	}
}

void WriteBackFence()
{
	_mm_sfence();
}

} // namespace hwcopy

void StoreFence()
{
	_mm_sfence();
}

} // namespace cmt
