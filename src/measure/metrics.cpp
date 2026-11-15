#include "common/aliases.h"
#include "measure/metrics.h"

namespace cmt {

namespace {
bool CpuAllowed(int cpu)
{
	if (cpu < 0 || cpu >= CPU_SETSIZE)
	{
		return false;
	}
	cpu_set_t s;
	CPU_ZERO(&s);
	if (sched_getaffinity(0, sizeof s, &s) != 0)
	{
		return true;
	}
	return CPU_ISSET(cpu, &s);
}

} // namespace

Result<void> Config::Validate() const
{
	if (!path)
	{
		Panic("Config::Validate without a path");
	}
	const PathSpec& p = *path;
	if (std::empty(sizes))
	{
		return Failure(Error::Kind::Usage, "no sizes");
	}
	for (auto b : sizes)
	{
		if (b % 16)
		{
			return Failure(Error::Kind::Usage,
						   "size is not a multiple of 16 B; the copy loops and the check move 8-16 B words");
		}
	}
	if (samples < 1)
	{
		return Failure(Error::Kind::Usage, "--samples must be >= 1");
	}
	if (!std::isfinite(budget_s) || budget_s < 0)
	{
		return Failure(Error::Kind::Usage, "--budget-s must be a finite number >= 0");
	}
	if (min_bytes == 0)
	{
		return Failure(Error::Kind::Usage, "--min-bytes must be > 0");
	}
	if (device < 0)
	{
		return Failure(Error::Kind::Usage, "--device must be >= 0");
	}

	for (Knob k : {Knob::Direction, Knob::Pages, Knob::Alloc, Knob::Fill, Knob::Threads, Knob::Slots,
				   Knob::SlotBytes, Knob::Wc, Knob::NoDma, Knob::PinStep, Knob::Unpin})
	{
		if (set_knobs.Has(k) && !p.accepts.Has(k))
		{
			return Failure(Error::Kind::Usage, "specified option has no effect on the selected --path");
		}
	}
	if (p.accepts.Has(Knob::Alloc) && alloc != Alloc::Register &&
		(set_knobs.Has(Knob::Pages) || pages != Pages::Small))
	{
		return Failure(Error::Kind::Usage, "--pages applies to --alloc register (mmap) only");
	}

	if (cpu >= 0)
	{
		const std::int64_t last = static_cast<std::int64_t>(cpu) +
								  (p.accepts.Has(Knob::Threads) ? static_cast<std::int64_t>(threads) - 1 : 0);
		if (last >= CPU_SETSIZE)
		{
			return Failure(Error::Kind::Usage, "--cpu and --threads exceed the supported affinity mask");
		}
		for (int c = cpu; c <= last; c++)
		{
			if (!CpuAllowed(c))
			{
				return Failure(Error::Kind::Usage, "requested CPU is not in this process's affinity mask");
			}
		}
	}
	else if (cpu != -1)
	{
		return Failure(Error::Kind::Usage, "--cpu must be >= 0 (or absent)");
	}
	if (p.validate)
	{
		CMT_RETURN_IF_ERROR(p.validate(*this));
	}
	if (p.dirs == PathSpec::Dirs::HtoDOnly && dir != Dir::HtoD)
	{
		return Failure(Error::Kind::Unsupported, "selected path is host to device only");
	}
	return {};
}

std::vector<std::size_t> SizeCoverage::Eligible() const
{
	std::vector<std::size_t> result;
	for (const auto& size : sizes)
	{
		if (size.limit == CapacityLimit::None)
		{
			result.push_back(size.bytes);
		}
	}
	return result;
}

SizeCoverage ComputeSizeCoverage(const std::vector<std::size_t>& requested, std::size_t device_available,
								 std::optional<std::size_t> host_available, unsigned device_multiplier,
								 unsigned host_multiplier, HostAllocationRequirements host_allocation)
{
	constexpr std::size_t margin = 256u << 20;
	const auto device_usable = device_available > margin ? device_available - margin : 0;
	const auto after_reserve = [](std::optional<std::size_t> available) -> std::optional<std::size_t>
	{
		return available
				   ? std::optional<std::size_t>(*available > margin ? (*available - margin) / 10 * 9 : 0)
				   : std::nullopt;
	};
	auto host_usable = after_reserve(host_available);
	const auto pinned_usable = host_allocation.separate_pinned_budget
								   ? after_reserve(host_allocation.pinned_available)
								   : host_usable;
	if (!host_allocation.separate_pinned_budget && host_usable)
	{
		*host_usable = *host_usable > host_allocation.fixed_pinned_bytes
						   ? *host_usable - host_allocation.fixed_pinned_bytes
						   : 0;
	}
	SizeCoverage result{
		device_available, device_usable, host_available, host_usable, device_multiplier, host_multiplier, {},
		host_allocation,  pinned_usable};
	for (const auto bytes : requested)
	{
		std::optional<std::size_t> caller_host_bytes = 0;
		if (host_multiplier)
		{
			const auto rounded = host_allocation.buffer_alignment
									 ? AlignUp(bytes, *host_allocation.buffer_alignment)
									 : std::nullopt;
			caller_host_bytes =
				rounded && *rounded <= std::numeric_limits<std::size_t>::max() / host_multiplier
					? std::optional<std::size_t>(*rounded * host_multiplier)
					: std::nullopt;
		}
		SizeCapacity size{bytes, CapacityLimit::None, caller_host_bytes};
		if (device_multiplier && bytes > device_usable / device_multiplier)
		{
			size.limit = CapacityLimit::Device;
			size.reason = "device_capacity";
		}
		else if (host_multiplier &&
				 (!host_allocation.buffer_alignment || *host_allocation.buffer_alignment == 0))
		{
			size.reason = "host_geometry_unknown";
		}
		else if (!caller_host_bytes)
		{
			size.reason = "host_allocation_overflow";
		}
		else if (host_multiplier && !host_usable)
		{
			size.reason = "host_capacity_unknown";
		}
		else if (host_allocation.fixed_pinned_bytes && !pinned_usable)
		{
			size.reason = "pinned_pool_capacity_unknown";
		}
		else if (host_allocation.fixed_pinned_bytes && host_allocation.fixed_pinned_bytes > *pinned_usable)
		{
			size.reason = "pinned_pool_capacity";
		}
		else if (host_multiplier && *caller_host_bytes > *host_usable)
		{
			size.reason = "host_capacity";
		}
		if (size.reason && size.limit == CapacityLimit::None)
		{
			size.limit = CapacityLimit::Host;
		}
		result.sizes.push_back(size);
	}
	return result;
}

std::string SizeCoverageJson(const SizeCoverage& coverage)
{
	JsonWriter writer;
	writer.start_object();
	writer.append_key_value("schema_version", 1);
	writer.append_comma();
	writer.escape_and_append_with_quotes("requested_sizes");
	writer.append_colon();
	writer.start_array();
	for (std::size_t i = 0; i < coverage.sizes.size(); ++i)
	{
		if (i != 0)
		{
			writer.append_comma();
		}
		writer.append(coverage.sizes[i].bytes);
	}
	writer.end_array();
	writer.append_comma();
	writer.escape_and_append_with_quotes("sizes");
	writer.append_colon();
	writer.start_array();
	std::size_t skipped = 0;
	for (std::size_t i = 0; i < coverage.sizes.size(); ++i)
	{
		if (i != 0)
		{
			writer.append_comma();
		}
		const auto& size = coverage.sizes[i];
		const char* reason = size.reason;
		skipped += reason != nullptr;
		writer.start_object();
		writer.append_key_value("size_bytes", size.bytes);
		JsonField(writer, "caller_host_bytes", size.caller_host_bytes);
		writer.append_comma();
		writer.append_key_value("status", reason ? "capacity_skipped" : "eligible");
		writer.append_comma();
		if (reason)
		{
			writer.append_key_value("reason", reason);
		}
		else
		{
			writer.append_key_value("reason", nullptr);
		}
		writer.end_object();
	}
	writer.end_array();
	writer.append_comma();
	writer.escape_and_append_with_quotes("device");
	writer.append_colon();
	writer.start_object();
	writer.append_key_value("available_bytes", coverage.device_available);
	writer.append_comma();
	writer.append_key_value("usable_bytes", coverage.device_usable);
	writer.append_comma();
	writer.append_key_value("multiplier", coverage.device_multiplier);
	writer.end_object();
	writer.append_comma();
	writer.escape_and_append_with_quotes("host");
	writer.append_colon();
	writer.start_object();
	if (coverage.host_available)
	{
		writer.append_key_value("available_bytes", *coverage.host_available);
	}
	else
	{
		writer.append_key_value("available_bytes", nullptr);
	}
	writer.append_comma();
	if (coverage.host_usable)
	{
		writer.append_key_value("usable_bytes", *coverage.host_usable);
	}
	else
	{
		writer.append_key_value("usable_bytes", nullptr);
	}
	writer.append_comma();
	writer.append_key_value("multiplier", coverage.host_multiplier);
	JsonField(writer, "buffer_alignment_bytes", coverage.host_allocation.buffer_alignment);
	JsonField(writer, "fixed_pinned_pool_bytes", coverage.host_allocation.fixed_pinned_bytes);
	JsonField(writer, "pinned_pool_budget",
			  coverage.host_allocation.separate_pinned_budget ? "separate_ordinary_memory"
															  : "shared_with_caller");
	JsonField(writer, "pinned_pool_available_bytes",
			  coverage.host_allocation.separate_pinned_budget ? coverage.host_allocation.pinned_available
															  : coverage.host_available);
	JsonField(writer, "pinned_pool_usable_bytes", coverage.pinned_usable);
	writer.append_comma();
	writer.append_key_value("basis", "host availability capped by applicable visible cgroup v2 limits");
	writer.append_comma();
	writer.append_key_value("limitation", "estimate only; concurrent use and hidden ancestors can reduce it");
	writer.end_object();
	writer.append_comma();
	writer.append_key_value("outcome", skipped == 0						  ? "eligible"
									   : skipped == coverage.sizes.size() ? "all_capacity_skipped"
																		  : "mixed");
	writer.end_object();
	std::string_view output;
	if (writer.view().get(output))
	{
		Panic("Cannot allocate size-coverage JSON output");
	}
	return std::string(output);
}

namespace {

bool IsPowerOfTwo(std::size_t x)
{
	return x && !(x & (x - 1));
}

std::string Trim(const std::string& text, std::size_t& begin)
{
	begin = 0;
	std::size_t end = text.size();
	while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])))
	{
		++begin;
	}
	while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])))
	{
		--end;
	}
	return text.substr(begin, end - begin);
}

template <std::size_t N>
MaybeError SizeError(const char (&message)[N], std::size_t offset, std::size_t* error_offset)
{
	if (error_offset != nullptr)
	{
		*error_offset = offset;
	}
	return UsageError(message);
}

} // namespace

Result<std::size_t> ParseSize(const std::string& raw, std::size_t* error_offset)
{
	std::size_t begin = 0;
	const std::string text = Trim(raw, begin);
	if (text.empty() || !std::isdigit(static_cast<unsigned char>(text[0])))
	{
		return SizeError("bad size", begin, error_offset);
	}
	std::size_t digit_offset = 0;
	unsigned long long number = 0;
	for (; digit_offset < text.size() && std::isdigit(static_cast<unsigned char>(text[digit_offset]));
		 ++digit_offset)
	{
		const unsigned digit = static_cast<unsigned>(text[digit_offset] - '0');
		if (number > (std::numeric_limits<unsigned long long>::max() - digit) / 10)
		{
			return SizeError("size overflows", begin + digit_offset, error_offset);
		}
		number = number * 10 + digit;
	}
	std::string suffix = text.substr(digit_offset);
	for (auto& character : suffix)
	{
		character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
	}
	unsigned shift = 0;
	if (!suffix.empty() && suffix != "B")
	{
		switch (suffix[0])
		{
		case 'K':
			shift = 10;
			break;
		case 'M':
			shift = 20;
			break;
		case 'G':
			shift = 30;
			break;
		default:
			return SizeError("bad size suffix (use K, M or G)",
							 begin + digit_offset + (suffix[0] == 'B' ? 1 : 0), error_offset);
		}
		const std::string rest = suffix.substr(1);
		if (!(std::empty(rest) || rest == "B" || rest == "IB"))
		{
			const std::string_view expected = rest[0] == 'I' ? "IB" : "B";
			std::size_t matched = 0;
			while (matched < rest.size() && matched < expected.size() && rest[matched] == expected[matched])
			{
				++matched;
			}
			return SizeError("bad size suffix", begin + digit_offset + 1 + matched, error_offset);
		}
	}
	if (number == 0)
	{
		return SizeError("size must be > 0", begin, error_offset);
	}
	if (shift && number > (std::numeric_limits<std::size_t>::max() >> shift))
	{
		return SizeError("size overflows", begin, error_offset);
	}
	const std::size_t bytes = static_cast<std::size_t>(number) << shift;
	if (bytes > kMaxBytes)
	{
		return SizeError("size is above the 1 TiB limit", begin, error_offset);
	}
	return bytes;
}

Result<std::vector<std::size_t>> Pow2Range(std::size_t lo, std::size_t hi)
{
	if (!IsPowerOfTwo(lo) || !IsPowerOfTwo(hi))
	{
		return Failure(Error::Kind::Usage, "range ends must be powers of two");
	}
	if (lo > hi)
	{
		return Failure(Error::Kind::Usage, "range start is larger than its end");
	}
	std::vector<std::size_t> v;
	for (std::size_t b = lo;; b <<= 1)
	{
		v.push_back(b);
		if (b == hi)
		{
			break;
		}
	}
	return v;
}

Result<std::vector<std::size_t>> ParseSizes(const std::string& text, std::size_t* error_offset)
{
	std::vector<std::size_t> sizes;
	std::size_t item_begin = 0;
	while (item_begin <= text.size())
	{
		std::size_t item_end = text.find(',', item_begin);
		if (item_end == std::string::npos)
		{
			item_end = text.size();
		}
		std::size_t leading_bytes = 0;
		const std::string item = text.substr(item_begin, item_end - item_begin);
		if (Trim(item, leading_bytes).empty())
		{
			return SizeError("empty item in size list", item_begin + leading_bytes, error_offset);
		}
		const std::size_t dots = item.find("..");
		if (dots == std::string::npos)
		{
			auto bytes = ParseSize(item, error_offset);
			if (!bytes)
			{
				if (error_offset != nullptr)
				{
					*error_offset += item_begin;
				}
				return cmt::Failure(bytes.error());
			}
			sizes.push_back(*bytes);
		}
		else
		{
			auto lo = ParseSize(item.substr(0, dots), error_offset);
			if (!lo)
			{
				if (error_offset != nullptr)
				{
					*error_offset += item_begin;
				}
				return cmt::Failure(lo.error());
			}
			auto hi = ParseSize(item.substr(dots + 2), error_offset);
			if (!hi)
			{
				if (error_offset != nullptr)
				{
					*error_offset += item_begin + dots + 2;
				}
				return cmt::Failure(hi.error());
			}
			auto range = Pow2Range(*lo, *hi);
			if (!range)
			{
				if (error_offset != nullptr)
				{
					std::size_t endpoint = !IsPowerOfTwo(*lo) ? 0 : dots + 2;
					while (endpoint < item.size() && std::isspace(static_cast<unsigned char>(item[endpoint])))
					{
						++endpoint;
					}
					*error_offset = item_begin + endpoint;
				}
				return cmt::Failure(range.error());
			}
			sizes.insert(std::end(sizes), std::begin(*range), std::end(*range));
		}
		item_begin = item_end + 1;
	}
	return sizes;
}

std::string HumanSize(std::size_t b)
{
	static const char* unit[] = {"B", "KiB", "MiB", "GiB", "TiB"};
	std::size_t u = 0;
	while (u < 4 && b >= 1024 && b % 1024 == 0)
	{
		b /= 1024, u++;
	}
	return fmt::format("{} {}", b, unit[u]);
}

std::uint32_t RepsFor(std::size_t bytes, std::size_t min_bytes)
{
	if (bytes == 0)
	{
		Panic("RepsFor: zero bytes");
	}
	const std::size_t r = min_bytes / bytes;
	return static_cast<std::uint32_t>(std::max<std::size_t>(1, std::min<std::size_t>(r, kMaxReps)));
}

Summary Summarize(std::vector<double> v)
{
	Summary s;
	if (std::empty(v))
	{
		return s;
	}
	std::sort(std::begin(v), std::end(v));
	s.n = std::size(v);
	s.min = v.front();
	s.best = v.back();
	s.median = v[std::size(v) / 2];
	return s;
}

double Gbps(double bytes, double seconds)
{
	return seconds > 0 ? bytes / seconds / 1e9 : 0.0;
}

namespace {

template <class F>
void ParallelFor(std::size_t n, std::size_t threads, F f)
{
	constexpr std::size_t kWordsPerWorker = 65536;
	const std::size_t count = std::max<std::size_t>(
		1, std::min<std::size_t>(std::max<std::size_t>(1, threads), n / kWordsPerWorker + 1));
	if (count == 1)
	{
		return f(std::size_t{0}, n);
	}
	std::vector<cmt::Thread> workers(count);
	for (std::size_t i = 0; i < count; ++i)
	{
		auto started = workers[i].Start([=] { f(n * i / count, n * (i + 1) / count); });
		if (!started)
		{
			for (; i < count; ++i)
			{
				f(n * i / count, n * (i + 1) / count);
			}
			break;
		}
	}
	for (auto& worker : workers)
	{
		if (!worker.Joinable())
		{
			continue;
		}
		auto joined = worker.Join();
		if (!joined)
		{
			Panic(joined.error().Message());
		}
	}
}

} // namespace

void PatternFill(void* p, std::size_t bytes, std::uint64_t seed, std::size_t threads)
{
	if (bytes % 8)
	{
		Panic("PatternFill: size is not a multiple of 8");
	}
	auto* w = static_cast<std::uint64_t*>(p);
	ParallelFor(bytes / 8, threads,
				[=](std::size_t a, std::size_t b)
				{
					for (std::size_t i = a; i < b; i++)
					{
						w[i] = PatternWord(i, seed);
					}
				});
}

std::size_t PatternCheck(const void* p, std::size_t bytes, std::uint64_t seed, std::size_t threads)
{
	if (bytes % 8)
	{
		Panic("PatternCheck: size is not a multiple of 8");
	}
	const auto* w = static_cast<const std::uint64_t*>(p);
	std::atomic<std::size_t> bad{0};
	constexpr std::size_t kReported = 8;
	struct Wrong
	{
		std::size_t index;
		std::uint64_t got;
	};
	std::atomic<std::size_t> found{0};
	Mutex reported_mutex;
	std::vector<Wrong> reported;
	ParallelFor(bytes / 8, threads,
				[&, w](std::size_t a, std::size_t b)
				{
					std::size_t x = 0;
					for (std::size_t i = a; i < b; i++)
					{
						const std::uint64_t got = w[i];
						if (got != PatternWord(i, seed))
						{
							x++;
							if (found.load(std::memory_order_relaxed) < kReported &&
								found.fetch_add(1, std::memory_order_relaxed) < kReported)
							{
								std::lock_guard<Mutex> guard(reported_mutex);
								reported.push_back({i, got});
							}
						}
					}
					bad += x;
				});
	std::sort(std::begin(reported), std::end(reported),
			  [](const Wrong& l, const Wrong& r) { return l.index < r.index; });
	for (const auto& wrong : reported)
	{
		const std::uint64_t want = PatternWord(wrong.index, seed);
		const std::uint64_t as_index = (wrong.got ^ seed) * 0xF1DE83E19937733Dull; // K^-1 mod 2^64
		fmt::print(stderr,
				   "pattern check: word {} (byte {:#x}) is {:#018x}, expected {:#018x}: {} bits differ; "
				   "it is this seed's word {}{}\n",
				   wrong.index, wrong.index * 8, wrong.got, want, __builtin_popcountll(wrong.got ^ want),
				   as_index < bytes / 8 ? fmt::to_string(as_index) : "of no index in range",
				   wrong.got == 0 ? " (zero)" : "");
	}
	return bad.load();
}

const char* AllocName(Alloc a)
{
	switch (a)
	{
	case Alloc::Register:
		return "register";
	case Alloc::HostAlloc:
		return "hostalloc";
	case Alloc::HostAllocWc:
		return "hostalloc_wc";
	}
	return "?";
}

const char* UnpinName(Unpin u)
{
	switch (u)
	{
	case Unpin::End:
		return "end";
	case Unpin::Eager:
		return "eager";
	case Unpin::Prereg:
		return "prereg";
	}
	return "?";
}

Result<Dir> ParseDir(const std::string& s)
{
	if (s == "htod" || s == "h2d")
	{
		return Dir::HtoD;
	}
	if (s == "dtoh" || s == "d2h")
	{
		return Dir::DtoH;
	}
	return Failure(Error::Kind::Usage, "unknown direction (htod|dtoh)");
}

Result<Alloc> ParseAlloc(const std::string& s)
{
	for (Alloc a : {Alloc::Register, Alloc::HostAlloc, Alloc::HostAllocWc})
	{
		if (s == AllocName(a))
		{
			return a;
		}
	}
	return Failure(Error::Kind::Usage, "unknown alloc (register|hostalloc|hostalloc_wc)");
}

Result<Unpin> ParseUnpin(const std::string& s)
{
	for (Unpin u : {Unpin::End, Unpin::Eager, Unpin::Prereg})
	{
		if (s == UnpinName(u))
		{
			return u;
		}
	}
	return Failure(Error::Kind::Usage, "unknown unpin (end|eager|prereg)");
}

namespace {
std::string QuoteCsvField(const std::string& text)
{
	if (text.find_first_of(",\"\r\n") == std::string::npos)
	{
		return text;
	}
	std::string quoted = "\"";
	for (char character : text)
	{
		if (character == '"')
		{
			quoted += '"';
		}
		quoted += character;
	}
	quoted += '"';
	return quoted;
}
} // namespace

const char* VerifiedName(Verified v)
{
	switch (v)
	{
	case Verified::Ok:
		return "ok";
	case Verified::Bad:
		return "bad";
	case Verified::Skip:
		return "skip";
	}
	return "?";
}

std::string CsvHeader()
{
	return "schema,tag,gpu,path,dir,pages,alloc,fill,threads,slots,slot_bytes,pin_step,unpin,flush,size_"
		   "bytes,reps,"
		   "samples,gbps_best,gbps_med,gbps_min,gbps_h2d,gbps_d2h,huge_kib,verified,dma,setup_s,copy_s,"
		   "teardown_s,"
		   "phase";
}

std::string CsvRow(const MeasurementRow& r)
{
	std::string s;
	auto add = [&s](const std::string& x)
	{
		if (!std::empty(s))
		{
			s += ',';
		}
		s += x;
	};
	add(fmt::to_string(kCsvSchema));
	add(QuoteCsvField(r.tag));
	add(QuoteCsvField(r.gpu));
	add(QuoteCsvField(r.path));
	add(QuoteCsvField(r.dir));
	add(QuoteCsvField(r.pages));
	add(QuoteCsvField(r.alloc));
	add(QuoteCsvField(r.fill));
	add(fmt::to_string(r.threads));
	add(fmt::to_string(r.slots));
	add(fmt::to_string(r.slot_bytes));
	add(fmt::to_string(r.pin_step));
	add(QuoteCsvField(r.unpin));
	add(r.flush ? "1" : "0");
	add(fmt::to_string(r.size_bytes));
	add(fmt::to_string(r.reps));
	add(fmt::to_string(r.samples));
	add(fmt::format("{:.3f}", r.gbps_best));
	add(fmt::format("{:.3f}", r.gbps_med));
	add(fmt::format("{:.3f}", r.gbps_min));
	add(fmt::format("{:.3f}", r.gbps_h2d));
	add(fmt::format("{:.3f}", r.gbps_d2h));
	add(r.huge_kib ? fmt::to_string(*r.huge_kib) : "");
	add(VerifiedName(r.verified));
	add(r.dma ? "1" : "0");
	add(fmt::format("{:.9g}", r.setup_s));
	add(fmt::format("{:.9g}", r.copy_s));
	add(fmt::format("{:.9g}", r.teardown_s));
	add(QuoteCsvField(r.phase));
	return s;
}

const char* DirColumn(const Config& c, const PathSpec& p)
{
	switch (p.dirs)
	{
	case PathSpec::Dirs::Both:
		return "both";
	case PathSpec::Dirs::None:
		return "-";
	default:
		return DirName(c.dir);
	}
}

MeasurementRow ResultIdentity(const Config& c, const DeviceInfo& dev, const PathRunner& runner, std::size_t b,
							  std::uint32_t reps, bool flushing, Verified verified)
{
	const PathSpec& p = *c.path;
	MeasurementRow base;
	base.tag = c.tag;
	base.gpu = dev.name;
	base.path = p.name;
	base.dir = DirColumn(c, p);
	base.pages = p.accepts.Has(Knob::Alloc) && c.alloc != Alloc::Register ? "-" : PagesName(c.pages);
	base.alloc = runner.Alloc();
	base.fill = p.accepts.Has(Knob::Fill) ? (c.cpu_copy ? LoopNameText(c.cpu_loop) : "none") : "-";
	base.unpin = p.accepts.Has(Knob::Unpin) ? UnpinName(c.unpin) : "-";
	base.threads = p.accepts.Has(Knob::Threads) ? c.threads : 1;
	base.slots = p.accepts.Has(Knob::Slots) ? c.slots : 0;
	base.slot_bytes = p.accepts.Has(Knob::SlotBytes) ? c.slot_bytes : 0;
	base.pin_step = p.accepts.Has(Knob::PinStep) ? c.pin_step : 0;
	base.flush = flushing;
	base.size_bytes = b;
	base.reps = reps;
	base.huge_kib = runner.HugeKiB();
	base.verified = verified;
	base.dma = !p.accepts.Has(Knob::NoDma) || c.dma;
	return base;
}

void SummarizeResult(MeasurementRow& row, const std::vector<SampleOut>& samples, unsigned moved_bytes)
{
	std::vector<double> rates;
	for (const auto& sample : samples)
	{
		rates.push_back(Gbps(static_cast<double>(row.size_bytes) * row.reps * moved_bytes, sample.seconds));
	}
	const Summary s = Summarize(rates);
	row.samples = s.n;
	row.gbps_best = s.best;
	row.gbps_med = s.median;
	row.gbps_min = s.min;
}

std::vector<MeasurementRow> PathRunner::Rows(const MeasurementRow& base, const std::vector<SampleOut>&) const
{
	return {base};
}

} // namespace cmt

namespace cmt {
namespace bench {
bool SchemaReporter::ReportContext(const Context&)
{
	if (header)
	{
		GetOutputStream() << cmt::CsvHeader() << '\n';
	}
	return true;
}

void SchemaReporter::ReportRuns(const std::vector<Run>& runs)
{
	for (const auto& run : runs)
	{
		if (run.run_type != Run::RT_Iteration)
		{
			continue;
		}
		auto& group = groups[run.run_name.function_name];
		if (run.skipped != benchmark::internal::NotSkipped || run.iterations != 1)
		{
			group.failed = true;
			continue;
		}
		bool missing_counter = false;
		const auto value = [&](const char* key)
		{
			const auto found = run.counters.find(key);
			if (found == run.counters.end())
			{
				missing_counter = true;
				return 0.0;
			}
			return found->second.value;
		};
		cmt::SampleOut sample;
		sample.seconds = run.real_accumulated_time;
		if (value("has_phases") != 0)
		{
			sample.phases = cmt::SampleOut::Phases{value("setup_s"), value("copy_s"), value("teardown_s")};
		}
		if (run.counters.count("has_lifecycle") && value("has_lifecycle") != 0)
		{
			sample.lifecycle.emplace();
			for (std::size_t i = 0; i < cmt::kLifecycleCounters.size(); i++)
			{
				(*sample.lifecycle)[i] = value(cmt::kLifecycleCounters[i]);
			}
		}
		if (value("has_directions") != 0)
		{
			sample.directions = cmt::SampleOut::Directions{value("htod_gbps"), value("dtoh_gbps")};
		}
		if (value("verified") == 0)
		{
			group.verified = cmt::Verified::Skip;
		}
		if (missing_counter)
		{
			group.failed = true;
			GetErrorStream() << "Incomplete sample counters: " << run.run_name.function_name << '\n';
			continue;
		}
		group.samples.push_back(sample);
	}
}

void SchemaReporter::Finalize()
{
	std::vector<cmt::MeasurementRow> rows;
	for (const auto& [name, group] : groups)
	{
		if (group.failed || std::empty(group.samples))
		{
			continue;
		}
		for (const auto& row : format_callback(name, group.samples, group.verified))
		{
			rows.push_back(row);
		}
	}
	std::stable_sort(std::begin(rows), std::end(rows),
					 [](const auto& a, const auto& b)
					 {
						 return std::tie(a.tag, a.path, a.dir, a.size_bytes) <
								std::tie(b.tag, b.path, b.dir, b.size_bytes);
					 });
	for (const auto& row : rows)
	{
		GetOutputStream() << cmt::CsvRow(row) << '\n';
	}
	GetOutputStream().flush();
}
} // namespace bench
} // namespace cmt

namespace cmt {

const char* TimingBoundary(const Config& config)
{
	const auto& path = config.path->name;
	if (path == "cpu_copy")
	{
		return "cpu_copy_only";
	}
	if (path == "pageable" || path == "staged")
	{
		if (path == "staged" && !config.cpu_copy)
		{
			return "dma_only";
		}
		return config.dma ? "completed_pageable_copy" : "cpu_copy_only";
	}
	if (path == "lifecycle")
	{
		return "host_device_lifecycle";
	}
	if (path == "host_allocation")
	{
		return "host_allocation_lifecycle";
	}
	if (path == "staged_lifecycle")
	{
		return config.cpu_copy ? "staging_lifecycle" : "staging_dma_lifecycle";
	}
	if (path == "register")
	{
		return "registration_lifecycle";
	}
	if (path == "pin_copy_unpin" || path == "pin_pipeline")
	{
		return "pin_copy_unpin";
	}
	if (path == "bidir_ce")
	{
		return "bidirectional_copy_only";
	}
	return "pinned_copy_only";
}

namespace {

void WritePhases(JsonWriter& writer, const SampleOut& sample, double divisor, const Config& config)
{
	JsonKey(writer, "phases_seconds");
	writer.start_object();
	if (sample.lifecycle)
	{
		const bool registered = config.alloc == Alloc::Register;
		const bool allocation_only = config.path->name == "host_allocation";
		const char* names[] = {registered ? "host_alloc_first_touch" : "host_alloc_pin_first_touch",
							   allocation_only ? nullptr : "device_alloc",
							   registered ? "pin" : nullptr,
							   allocation_only ? nullptr : "copy",
							   registered ? "unpin" : nullptr,
							   allocation_only ? nullptr : "device_free",
							   registered ? "host_free" : "host_unpin_free"};
		for (std::size_t i = 0; i < sample.lifecycle->size(); ++i)
		{
			if (names[i])
			{
				JsonField(writer, names[i], (*sample.lifecycle)[i] / divisor, i == 0);
			}
		}
	}
	else if (sample.phases)
	{
		JsonField(writer, "setup", sample.phases->setup / divisor, true);
		JsonField(writer, "copy", sample.phases->copy / divisor);
		JsonField(writer, "teardown", sample.phases->teardown / divisor);
	}
	writer.end_object();
}

bool ValidVmaPages(const SampleOut::VmaPages& pages, std::size_t allocations)
{
	if (pages.allocations != allocations || pages.observed_allocations > allocations)
	{
		return false;
	}
	if (pages.observed_allocations == 0)
	{
		return !pages.min_huge_kib && !pages.max_huge_kib;
	}
	return pages.min_huge_kib && pages.max_huge_kib && *pages.min_huge_kib >= 0 &&
		   *pages.max_huge_kib >= *pages.min_huge_kib;
}

void WriteVmaPages(JsonWriter& writer, const char* name, const SampleOut::VmaPages& pages)
{
	JsonKey(writer, name);
	writer.start_object();
	JsonField(writer, "allocations", pages.allocations, true);
	JsonField(writer, "observed_allocations", pages.observed_allocations);
	JsonField(writer, "min_huge_kib", pages.min_huge_kib);
	JsonField(writer, "max_huge_kib", pages.max_huge_kib);
	writer.end_object();
}

void WriteHostBacking(JsonWriter& writer, const SampleOut::HostBacking& backing)
{
	JsonKey(writer, "host_backing");
	writer.start_object();
	JsonField(writer, "basis", "smaps totals for overlapping VMAs; not exact payload coverage", true);
	WriteVmaPages(writer, "before_pin", backing.before_pin);
	WriteVmaPages(writer, "after_pin", backing.after_pin);
	writer.end_object();
}

void WriteCpuBacking(JsonWriter& writer, const SampleOut::CpuBacking& backing)
{
	JsonKey(writer, "cpu_backing");
	writer.start_object();
	JsonField(writer, "scope", "whole_overlapping_pageable_vmas;cuda_owned_unqueried", true);
	WriteVmaPages(writer, "source_before", backing.source_before);
	WriteVmaPages(writer, "source_after", backing.source_after);
	WriteVmaPages(writer, "destination_before", backing.destination_before);
	WriteVmaPages(writer, "destination_after", backing.destination_after);
	writer.end_object();
}

void WriteMeasurementSettings(JsonWriter& writer, const Measurement& measurement)
{
	const auto& config = measurement.config;
	JsonKey(writer, "settings");
	if (!measurement.settings_json.empty())
	{
		writer.append_raw(measurement.settings_json);
		return;
	}
	writer.start_object();
	const bool staged = config.path->name == "staged" || config.path->name == "staged_lifecycle";
	JsonField(writer, "method",
			  staged ? "staged" : (measurement.execution == "hook_unverified" ? "cuda_api" : "driver"), true);
	if (staged)
	{
		JsonField(writer, "cpu_loop", config.cpu_copy ? LoopNameText(config.cpu_loop) : "none");
		JsonField(writer, "pipeline_workers", config.threads);
		JsonField(writer, "slots_per_worker", config.slots);
		JsonField(writer, "chunk_bytes", config.slot_bytes);
		JsonField(writer, "staging_memory", config.wc_slots ? "wc_pinned" : "cached_pinned");
	}
	if (config.path->name == "lifecycle" || config.path->name == "host_allocation")
	{
		JsonField(writer, "allocation", AllocName(config.alloc));
		JsonField(writer, "first_touch", "write one byte per native base page");
		JsonField(writer, "base_page_bytes", BasePageBytes());
	}
	writer.end_object();
}

const char* CachePolicy(const Measurement& measurement)
{
	const auto& config = measurement.config;
	if (config.path->dirs == PathSpec::Dirs::None && measurement.direction_name.empty())
	{
		return "not_applicable";
	}
	if (config.path->name == "pinned_ce" && config.alloc == Alloc::HostAllocWc)
	{
		return "wc_unflushed";
	}
	if (!config.flush || !FlushSupported())
	{
		return "warm";
	}
	return measurement.copies_per_sample == 1 ? "evicted_before_sample" : "evicted_before_first_copy";
}

Result<void> WriteMeasurement(JsonWriter& writer, const Measurement& measurement)
{
	if (!measurement.config.path || measurement.bytes == 0 || measurement.copies_per_sample == 0)
	{
		return UsageError("measurement has invalid identity");
	}
	const bool cpu_copy = measurement.config.path->name == "cpu_copy";
	bool source_pageable = false;
	bool destination_pageable = false;
	if (cpu_copy && measurement.settings_json.empty())
	{
		return UsageError("CPU copy requires explicit endpoint settings");
	}
	if (!measurement.settings_json.empty())
	{
		JsonParser parser;
		Json settings;
		if (parser.parse(measurement.settings_json).get(settings) || !settings.is_object())
		{
			return UsageError("measurement settings must be a JSON object");
		}
		if (cpu_copy)
		{
			std::string_view source;
			std::string_view destination;
			std::string_view pages;
			const auto known_memory = [](std::string_view memory)
			{ return memory == "pageable" || memory == "pinned_cached" || memory == "pinned_wc"; };
			if (settings["source_memory"].get_string().get(source) ||
				settings["destination_memory"].get_string().get(destination) ||
				settings["pageable_pages"].get_string().get(pages) || !known_memory(source) ||
				!known_memory(destination) ||
				(measurement.config.pages != Pages::Small && measurement.config.pages != Pages::Thp) ||
				pages != PagesName(measurement.config.pages) || measurement.pages_requested != pages)
			{
				return UsageError("CPU copy endpoint or page-policy settings are inconsistent");
			}
			source_pageable = source == "pageable";
			destination_pageable = destination == "pageable";
		}
	}
	const auto& config = measurement.config;
	writer.start_object();
	const auto default_direction =
		config.path->dirs == PathSpec::Dirs::None
			? "none"
			: (config.path->dirs == PathSpec::Dirs::Both ? "both" : DirName(config.dir));
	const auto direction =
		measurement.direction_name.empty() ? default_direction : measurement.direction_name;
	JsonField(writer, "direction", direction, true);
	JsonField(writer, "bytes", measurement.bytes);
	JsonField(writer, "process_round", measurement.process_round);
	JsonField(writer, "path", config.path->name);
	JsonField(writer, "tag", config.tag);
	JsonField(writer, "status", measurement.status);
	JsonField(writer, "reason", measurement.reason);
	JsonField(writer, "execution", measurement.execution);
	JsonField(writer, "coverage", measurement.coverage);
	JsonField(writer, "measurement_class", measurement.measurement_class);
	if (!measurement.experiment.empty())
	{
		JsonKey(writer, "experiment");
		writer.start_object();
		bool first = true;
		for (const auto& [name, value] : measurement.experiment)
		{
			JsonField(writer, name, value, first);
			first = false;
		}
		writer.end_object();
	}
	JsonField(writer, "timing_boundary", TimingBoundary(config));
	JsonField(writer, "verified", measurement.verified);
	JsonField(writer, "verification", measurement.verified ? measurement.verification : "none");
	JsonField(writer, "cache_policy", CachePolicy(measurement));
	const bool driver_pages = config.path->accepts.Has(Knob::Alloc) && config.alloc != Alloc::Register;
	JsonField(writer, "pages_requested",
			  measurement.pages_requested.empty() ? (driver_pages ? "driver" : PagesName(config.pages))
												  : measurement.pages_requested);
	JsonField(writer, "huge_kib", measurement.huge_kib);
	JsonField(writer, "allocation", measurement.allocation);
	JsonField(writer, "moved_bytes_multiplier", config.path->moved_bytes);
	WriteMeasurementSettings(writer, measurement);
	JsonField(writer, "sample_count", measurement.samples.size());
	const double copies = static_cast<double>(measurement.copies_per_sample);
	if (measurement.status == "complete")
	{
		if (measurement.samples.empty())
		{
			return UsageError("complete measurement has no samples");
		}
		std::vector<std::size_t> order;
		for (std::size_t i = 0; i < measurement.samples.size(); ++i)
		{
			const auto& sample = measurement.samples[i];
			if (!std::isfinite(sample.seconds) || sample.seconds <= 0)
			{
				return UsageError("sample duration must be finite and positive");
			}
			if ((config.path->name == "lifecycle" || config.path->name == "host_allocation") &&
				!sample.host_backing)
			{
				return UsageError("allocation lifecycle sample lacks host backing observations");
			}
			if (sample.host_backing)
			{
				for (const auto* pages : {&sample.host_backing->before_pin, &sample.host_backing->after_pin})
				{
					if (!ValidVmaPages(*pages, measurement.copies_per_sample))
					{
						return UsageError("host backing observation does not describe every allocation");
					}
				}
			}
			if (cpu_copy && !sample.cpu_backing)
			{
				return UsageError("CPU copy sample lacks endpoint backing observations");
			}
			if (sample.cpu_backing)
			{
				const auto& backing = *sample.cpu_backing;
				for (const auto& [pages, pageable] :
					 {std::pair{&backing.source_before, source_pageable},
					  std::pair{&backing.source_after, source_pageable},
					  std::pair{&backing.destination_before, destination_pageable},
					  std::pair{&backing.destination_after, destination_pageable}})
				{
					if (!cpu_copy || !ValidVmaPages(*pages, 1) ||
						(!pageable && pages->observed_allocations != 0) ||
						(pages->observed_allocations != 0 && pages->min_huge_kib != pages->max_huge_kib))
					{
						return UsageError(
							"CPU backing must describe one allocation; CUDA-owned endpoints stay unqueried");
					}
				}
			}
			order.push_back(i);
		}
		std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right)
						 { return measurement.samples[left].seconds < measurement.samples[right].seconds; });
		const auto index = order[order.size() / 2];
		const auto& sample = measurement.samples[index];
		JsonField(writer, "median_sample_index", index);
		JsonField(writer, "seconds", sample.seconds / copies);
		if (config.path->name != "host_allocation")
		{
			JsonField(writer, "gbps",
					  Gbps(static_cast<double>(measurement.bytes) * config.path->moved_bytes,
						   sample.seconds / copies));
		}
		WritePhases(writer, sample, copies, config);
		if (sample.host_backing)
		{
			WriteHostBacking(writer, *sample.host_backing);
		}
		if (sample.cpu_backing)
		{
			WriteCpuBacking(writer, *sample.cpu_backing);
		}
	}
	JsonKey(writer, "samples");
	writer.start_array();
	for (std::size_t i = 0; i < measurement.samples.size(); ++i)
	{
		if (i)
		{
			writer.append_comma();
		}
		const auto& sample = measurement.samples[i];
		writer.start_object();
		JsonField(writer, "seconds", sample.seconds / copies, true);
		JsonField(writer, "elapsed_seconds", sample.seconds);
		JsonField(writer, "copies_per_sample", measurement.copies_per_sample);
		WritePhases(writer, sample, copies, config);
		if (sample.host_backing)
		{
			WriteHostBacking(writer, *sample.host_backing);
		}
		if (sample.cpu_backing)
		{
			WriteCpuBacking(writer, *sample.cpu_backing);
		}
		writer.end_object();
	}
	writer.end_array();
	for (const auto& [key, value] : measurement.observations)
	{
		JsonField(writer, key, value);
	}
	writer.end_object();
	return {};
}

} // namespace

Result<void> WriteMeasurements(const std::string& path, const std::string& source_machine,
							   const std::vector<Measurement>& measurements,
							   const std::map<std::string, double>& timings_seconds,
							   const std::string& diagnostics)
{
	JsonParser parser;
	Json machine;
	if (parser.parse(source_machine).get(machine) || !machine.is_object())
	{
		return UsageError("source_machine must be a JSON object");
	}
	if (!diagnostics.empty())
	{
		Json value;
		if (parser.parse(diagnostics).get(value) || !value.is_object())
		{
			return UsageError("measurement diagnostics must be a JSON object");
		}
	}
	JsonWriter writer;
	writer.start_object();
	JsonField(writer, "schema", "cmt.measurements", true);
	JsonField(writer, "version", 1);
	JsonKey(writer, "source_machine");
	writer.append_raw(source_machine);
	if (!diagnostics.empty())
	{
		JsonKey(writer, "diagnostics");
		writer.append_raw(diagnostics);
	}
	if (!timings_seconds.empty())
	{
		JsonKey(writer, "timings_seconds");
		writer.start_object();
		bool first = true;
		for (const auto& [name, seconds] : timings_seconds)
		{
			if (!std::isfinite(seconds) || seconds < 0)
			{
				return UsageError("process phase duration must be finite and nonnegative");
			}
			JsonField(writer, name, seconds, first);
			first = false;
		}
		writer.end_object();
	}
	JsonKey(writer, "runs");
	writer.start_array();
	for (std::size_t i = 0; i < measurements.size(); ++i)
	{
		if (i)
		{
			writer.append_comma();
		}
		CMT_RETURN_IF_ERROR(WriteMeasurement(writer, measurements[i]));
	}
	writer.end_array();
	writer.end_object();
	auto text = FinishJson(writer);
	if (!text)
	{
		return Failure(text.error());
	}
	return WriteTextFileAtomic(path, *text);
}

} // namespace cmt
