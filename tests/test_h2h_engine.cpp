// test_h2h_engine.cpp - the h2h replacement body on synthetic copy
// descriptors laid out per a versions_db row (all ten x86-64 and all four
// aarch64 rows, on any host): op_kind_3, op_kind_1 and
// op_kind_2 operands, x/y/z offsets, widths, 3D sizes, and the BENCH_DMA guard.
// CTest covers serial and worker-pool copies, the MPMC protocol, and the DMA-only guard.
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "check.h"
#include "config.h"
#include "hook.h"
#include "cuda/versions.h"

namespace {

using cmt::patch::DriverVersion;

void wr64(uint8_t* p, uint16_t off, uint64_t v)
{
	std::memcpy(p + off, &v, 8);
}

void wr32(uint8_t* p, uint16_t off, uint32_t v)
{
	std::memcpy(p + off, &v, 4);
}

uint64_t addr(const void* p)
{
	return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
}

enum class Kind : std::uint8_t
{
	OpKind3,
	OpKind1,
	OpKind2
};

struct Surface
{
	uint8_t* base;
	uint64_t pitch;
	uint64_t height; // rows per slice
	uint64_t unkx, unky, unkz;
};

// Writes one operand at op (the src operand, or src + op_stride).
void write_operand(const DriverVersion& vi, uint8_t* op, Kind kind, const Surface& s, uint8_t* pointee1)
{
	wr64(op, vi.op_unkx, s.unkx);
	wr64(op, vi.op_unky, s.unky);
	wr64(op, vi.op_unkz, s.unkz);
	switch (kind)
	{
	case Kind::OpKind3:
		wr32(op, vi.op_kind, vi.op_kind_3);
		wr64(op, vi.op_some_pointer2, addr(s.base));
		wr64(op, vi.op_some_pointer2_width, s.pitch);
		wr64(op, vi.op_some_pointer2_height, s.height);
		break;
	case Kind::OpKind1:
		// pointee1->op_some_pointer1_address + op_some_pointer1_off = base; split it to exercise the add.
		wr32(op, vi.op_kind, vi.op_kind_1);
		wr64(pointee1, vi.op_some_pointer1_address, addr(s.base) - 64);
		wr64(op, vi.op_some_pointer1, addr(pointee1));
		wr64(op, vi.op_some_pointer1_off, 64);
		wr64(op, vi.op_some_pointer1_width, s.pitch);
		wr64(op, vi.op_some_pointer1_height, s.height);
		break;
	case Kind::OpKind2:
		wr32(op, vi.op_kind, vi.op_kind_2);
		wr64(op, vi.op_some_pointer3, addr(s.base));
		wr64(op, vi.op_some_pointer3_width, s.pitch);
		wr64(op, vi.op_some_pointer3_height, s.height);
		break;
	}
}

uint8_t* at(const Surface& s, uint64_t x, uint64_t y, uint64_t z)
{
	return s.base + ((s.unkz + z) * s.height + s.unky + y) * s.pitch + s.unkx + x;
}

void run_case(const DriverVersion& vi, Kind sk, Kind dk, uint64_t w, uint64_t h, uint64_t d)
{
	const Surface src_s{nullptr, w + 96, h + 5, 17, 2, 1};
	const Surface dst_s{nullptr, w + 40, h + 3, 9, 1, 0};
	check::Buffer src(src_s.pitch * src_s.height * (d + 2) + 64);
	check::Buffer dst(dst_s.pitch * dst_s.height * (d + 2) + 64, 0xEE);
	check::Buffer expect(std::size(dst), 0xEE);
	Surface s = src_s;
	Surface t = dst_s;
	s.base = std::data(src) + 64;
	t.base = std::data(dst) + 64;
	for (std::size_t i = 0; i < std::size(src); i++)
	{
		src[i] = check::pattern(i, 11);
	}
	Surface e = t;
	e.base = std::data(expect) + 64;
	for (uint64_t z = 0; z < d; z++)
	{
		for (uint64_t y = 0; y < h; y++)
		{
			std::memcpy(at(e, 0, y, z), at(s, 0, y, z), w);
		}
	}

	check::Buffer desc(0x400);
	check::Buffer src_pointee1(0x100);
	check::Buffer dst_pointee1(0x100);
	write_operand(vi, std::data(desc), sk, s, std::data(src_pointee1));
	write_operand(vi, std::data(desc) + vi.op_stride, dk, t, std::data(dst_pointee1));
	wr64(std::data(desc), vi.copying_width_bytes, w);
	wr64(std::data(desc), vi.copying_height_bytes, h);
	wr64(std::data(desc), vi.copying_depth_bytes, d);

	cmt::driver::CopyHostDescriptor(vi, std::data(desc));
	const bool staged = sk == Kind::OpKind1 || dk == Kind::OpKind1;
	if (cmt::GetConfig().bench_dma && staged)
	{
		check::Buffer untouched(std::size(dst), 0xEE);
		CHECKF(std::memcmp(std::data(dst), std::data(untouched), std::size(dst)) == 0, "%s bench_dma copied",
			   vi.version);
		return;
	}
	CHECKF(std::memcmp(std::data(dst), std::data(expect), std::size(dst)) == 0,
		   "%s kinds %d->%d w=%llu h=%llu d=%llu", vi.version, static_cast<int>(sk), static_cast<int>(dk),
		   (unsigned long long)w, (unsigned long long)h, (unsigned long long)d);
}

void test_null_operand(const DriverVersion& vi)
{
	check::Buffer desc(0x400);
	wr32(std::data(desc), vi.op_kind, vi.op_kind_1); // op_some_pointer1 0 -> no host pointer
	wr32(std::data(desc) + vi.op_stride, vi.op_kind, vi.op_kind_3);
	wr64(std::data(desc), vi.copying_width_bytes, 100);
	wr64(std::data(desc), vi.copying_height_bytes, 1);
	wr64(std::data(desc), vi.copying_depth_bytes, 1);
	cmt::driver::CopyHostDescriptor(vi, std::data(desc)); // must return without touching memory
	CHECK(true);
}

} // namespace

int main()
{
	// A failed cleanup call is reported and returns false (the caller retains the resource); it never terminates
	// the process, which may be inside the application's own teardown after a sticky CUDA error (driver.h).
	CHECK(cmt::cu::CheckCleanupResult(cmt::cu::kSuccess, "host-only success fixture"));
	CHECK(!cmt::cu::CheckCleanupResult(cmt::cu::kErrorNotSupported, "host-only failure fixture"));
	// Every row of both architectures: the descriptor port is plain C++ over the row's offsets, the same on either.
	std::vector<const DriverVersion*> rows;
	for (const cmt::Arch arch : {cmt::Arch::X86_64, cmt::Arch::Aarch64})
	{
		const cmt::patch::VersionTable& t = cmt::patch::VersionTableFor(arch);
		CHECK(t.count > 0);
		for (std::size_t i = 0; i < t.count; i++)
		{
			rows.push_back(&t.rows[i]);
		}
	}
	for (const DriverVersion* row : rows)
	{
		const DriverVersion& vi = *row;
		const Kind kinds[] = {Kind::OpKind3, Kind::OpKind1, Kind::OpKind2};
		for (Kind sk : kinds)
		{
			for (Kind dk : kinds)
			{
				run_case(vi, sk, dk, 1000, 1, 1);
				run_case(vi, sk, dk, 333, 7, 3);
			}
		}
		// Large enough to split in the pool: one row, and many rows.
		run_case(vi, Kind::OpKind3, Kind::OpKind1, (2u << 20) + 77, 1, 1);
		run_case(vi, Kind::OpKind1, Kind::OpKind3, 4096 + 3, 400, 2);
		test_null_operand(vi);
	}
	return check::exit_code();
}
