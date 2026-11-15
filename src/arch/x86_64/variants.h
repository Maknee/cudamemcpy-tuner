#pragma once

#define CMT_X86_64_COPY_VARIANTS(X)                                                                          \
	X(Memcpy, Memcpy, Base, CopyMemcpy)                                                                      \
	X(Movsb, Movsb, Base, CopyMovsb)                                                                         \
	X(CachedSse2, Cached, Sse2, CopyCachedSse2)                                                              \
	X(CachedAvx2, Cached, Avx2, CopyCachedAvx2)                                                              \
	X(CachedAvx512, Cached, Avx512, CopyCachedAvx512)                                                        \
	X(NtStoreSse2, NtStore, Sse2, CopyNtStoreSse2)                                                           \
	X(NtStoreAvx2, NtStore, Avx2, CopyNtStoreAvx2)                                                           \
	X(NtStoreAvx512, NtStore, Avx512, CopyNtStoreAvx512)                                                     \
	X(NtLoadAvx2, NtLoad, Avx2, CopyNtLoadAvx2)                                                              \
	X(NtLoadAvx512, NtLoad, Avx512, CopyNtLoadAvx512)                                                        \
	X(NtStorePrefetchSse2, NtStorePrefetch, Sse2, CopyNtStorePrefetchSse2)                                   \
	X(NtStorePrefetchAvx2, NtStorePrefetch, Avx2, CopyNtStorePrefetchAvx2)                                   \
	X(NtStorePrefetchAvx512, NtStorePrefetch, Avx512, CopyNtStorePrefetchAvx512)
