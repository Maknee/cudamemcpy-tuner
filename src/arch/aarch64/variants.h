#pragma once

#define CMT_AARCH64_COPY_VARIANTS(X)                                                                         \
	X(Memcpy, Memcpy, Base, CopyMemcpy)                                                                      \
	X(CachedNeon, Cached, Neon, CopyCachedNeon)                                                              \
	X(CachedSve, Cached, Sve, CopyCachedSve)                                                                 \
	X(NtStoreNeon, NtStore, Neon, CopyNtStoreNeon)                                                           \
	X(NtStoreSve, NtStore, Sve, CopyNtStoreSve)                                                              \
	X(NtLoadNeon, NtLoad, Neon, CopyNtLoadNeon)                                                              \
	X(NtLoadSve, NtLoad, Sve, CopyNtLoadSve)
