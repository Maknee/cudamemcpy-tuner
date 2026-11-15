#include "remote.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <system_error>

#include <fmt/format.h>
#include <picosha2.h>

#include "arch/copy.h"
#include "common/helpers.h"
#include "runner.h"
#include "share.h"

namespace cmt {
namespace {

class Scratch
{
public:
	static Result<Scratch> Create(const std::string& cache)
	{
		std::string pattern = cache + "/.download-XXXXXX";
		if (!mkdtemp(pattern.data()))
		{
			return OsError("create the download directory");
		}
		return Scratch(std::move(pattern));
	}

	Scratch(Scratch&& other) noexcept : path_(std::move(other.path_))
	{
		other.path_.clear();
	}

	Scratch(const Scratch&) = delete;
	Scratch& operator=(const Scratch&) = delete;
	Scratch& operator=(Scratch&&) = delete;

	~Scratch()
	{
		if (!std::empty(path_))
		{
			std::error_code
				ignored;
			std::filesystem::remove_all(path_, ignored);
		}
	}

	std::string File(std::string_view name) const
	{
		return path_ + "/" + std::string(name);
	}

private:
	explicit Scratch(std::string path) : path_(std::move(path))
	{
	}

	std::string path_;
};

struct Response
{
	int status = 0;
	std::string body;
	std::string headers;
};

bool EqualsIgnoringCase(std::string_view a, std::string_view b)
{
	return std::size(a) == std::size(b) &&
		   std::equal(std::begin(a), std::end(a), std::begin(b),
					  [](char x, char y)
					  {
						  return std::tolower(static_cast<unsigned char>(x)) ==
								 std::tolower(static_cast<unsigned char>(y));
					  });
}

struct UrlParts
{
	std::string_view scheme;
	std::optional<std::string_view> userinfo;
	std::string_view hostport;
	std::string_view rest;
};

std::optional<UrlParts> SplitUrl(std::string_view url)
{
	const auto separator = url.find("://");
	if (separator == std::string_view::npos || separator == 0)
	{
		return std::nullopt;
	}
	UrlParts parts;
	parts.scheme = url.substr(0, separator);
	auto authority = url.substr(separator + 3);
	const auto end = authority.find_first_of("/?#");
	if (end != std::string_view::npos)
	{
		parts.rest = authority.substr(end);
		authority = authority.substr(0, end);
	}
	if (const auto at = authority.rfind('@'); at != std::string_view::npos)
	{
		parts.userinfo = authority.substr(0, at);
		authority = authority.substr(at + 1);
	}
	parts.hostport = authority;
	return parts;
}

std::string Redacted(std::string_view url)
{
	const auto parts = SplitUrl(url);
	if (!parts || !parts->userinfo)
	{
		return std::string(url);
	}
	return fmt::format("{}://***@{}{}", parts->scheme, parts->hostport, parts->rest);
}

const char* CurlReason(int exit_code)
{
	switch (exit_code)
	{
	case 6:
		return "cannot resolve the host";
	case 7:
		return "cannot connect";
	case 28:
		return "timed out";
	case 35:
	case 51:
	case 58:
	case 60:
		return "TLS failed";
	case 47:
		return "too many redirects";
	case 63:
		return "the response exceeds the size limit";
	default:
		return "see curl's message above";
	}
}

bool LocalHttp(std::string_view url)
{
	const auto parts = SplitUrl(url);
	if (!parts || !EqualsIgnoringCase(parts->scheme, "http") || parts->userinfo)
	{
		return false;
	}
	std::string_view host = parts->hostport;
	std::string_view port;
	if (!std::empty(host) && host.front() == '[')
	{
		const auto close = host.find(']');
		if (close == std::string_view::npos)
		{
			return false;
		}
		port = host.substr(close + 1);
		host = host.substr(0, close + 1);
	}
	else if (const auto colon = host.find(':'); colon != std::string_view::npos)
	{
		port = host.substr(colon);
		host = host.substr(0, colon);
	}
	if (!std::empty(port) &&
		(port.front() != ':' || std::size(port) > 6 ||
		 !std::all_of(std::begin(port) + 1, std::end(port), [](char c) { return c >= '0' && c <= '9'; })))
	{
		return false;
	}
	return EqualsIgnoringCase(host, "localhost") || host == "127.0.0.1" || host == "[::1]";
}

Result<void> CheckUrl(std::string_view url)
{
	const bool printable =
		std::all_of(std::begin(url), std::end(url), [](char c) { return c > ' ' && c < 0x7f; });
	const auto parts = SplitUrl(url);
	if (!printable || !parts || std::empty(parts->hostport) ||
		(!EqualsIgnoringCase(parts->scheme, "https") && !LocalHttp(url)))
	{
		return UsageError(
			"a tune URL must be https:// (http:// only to localhost, without a user name), without "
			"spaces");
	}
	return {};
}

Result<std::string> CacheDirectory()
{
	auto directory = CacheDirectoryPath();
	if (!directory)
	{
		return Failure(directory.error());
	}
	CMT_RETURN_IF_ERROR(MakeDirectories(*directory));
	return directory;
}

Result<Response> Get(const std::string& url, const Scratch& scratch)
{
	fmt::print(stderr, "cmt: GET {}\n", Redacted(url));
	const auto body = scratch.File("body");
	const auto headers = scratch.File("headers");
	ProcessCommand curl;
	curl.executable = "curl";
	curl.arguments = {"-q",
					  "--silent",
					  "--show-error",
					  "--globoff",
					  "--proto",
					  LocalHttp(url) ? "=http,https" : "=https",
					  "--proto-redir",
					  "=https",
					  "--location",
					  "--max-redirs",
					  "5",
					  "--connect-timeout",
					  "10",
					  "--max-time",
					  "60",
					  "--max-filesize",
					  std::to_string(share::kMaxShareBytes),
					  "--user-agent",
					  "cmt",
					  "--output",
					  body,
					  "--dump-header",
					  headers,
					  "--write-out",
					  "%{http_code}\\n",
					  "--url",
					  url};
	curl.stdout_path = scratch.File("status");
	curl.stderr_path = scratch.File("stderr");
	auto finished = RunProcess(curl);
	if (!finished)
	{
		if (finished.error().GetCode() == ENOENT)
		{
			fmt::print(stderr, "cmt: curl is not on PATH; install it to download tunes\n");
		}
		return Failure(finished.error());
	}
	if (!finished->completed || finished->exit_code != 0)
	{
		auto message = ReadTextFile(curl.stderr_path, 64 << 10);
		fmt::print(stderr, "{}cmt: curl exited {}: {}\n", message ? *message : "", finished->exit_code,
				   CurlReason(finished->exit_code));
		return Failure(Error::Kind::Os, "download failed");
	}
	auto status = ReadTextFile(curl.stdout_path, 4096);
	if (!status)
	{
		return Failure(status.error());
	}
	Response response;
	const auto end = status->data() + std::min(status->find('\n'), std::size(*status));
	if (std::from_chars(status->data(), end, response.status).ptr != end || response.status <= 0)
	{
		return Failure(Error::Kind::Os, "curl reported no HTTP status");
	}
	for (auto [path, into] : {std::pair{body, &response.body}, std::pair{headers, &response.headers}})
	{
		auto text = ReadTextFile(path, share::kMaxShareBytes);
		if (!text && text.error().GetCode() != ENOENT)
		{
			return Failure(text.error());
		}
		*into = text ? std::move(*text) : std::string();
	}
	return response;
}

Result<std::string> Keep(const std::string& cache, const Scratch& scratch, const std::string& url,
						 const std::string& text)
{
	const auto path = cache + "/" + picosha2::hash256_hex_string(url).substr(0, 16) + ".json";
	const auto written = scratch.File("checked.json");
	CMT_RETURN_IF_ERROR(WriteTextFile(written, text));
	std::error_code error;
	std::filesystem::rename(written, path, error);
	if (error)
	{
		return IoFailure("rename the downloaded tune into the cache", error);
	}
	return path;
}

} // namespace

bool IsUrl(std::string_view text)
{
	const auto scheme = text.substr(0, text.find("://"));
	return std::size(scheme) < std::size(text) &&
		   (EqualsIgnoringCase(scheme, "https") || EqualsIgnoringCase(scheme, "http"));
}

std::string UrlEncode(std::string_view text)
{
	std::string encoded;
	for (char c : text)
	{
		const auto byte = static_cast<unsigned char>(c);
		if ((byte < 0x80 && std::isalnum(byte)) || c == '-' || c == '.' || c == '_' || c == '~')
		{
			encoded += c;
		}
		else
		{
			encoded += fmt::format("%{:02X}", static_cast<unsigned>(byte));
		}
	}
	return encoded;
}

Result<std::string> DownloadShare(const std::string& url)
{
	CMT_RETURN_IF_ERROR(CheckUrl(url));
	auto cache = CacheDirectory();
	if (!cache)
	{
		return Failure(cache.error());
	}
	auto scratch = Scratch::Create(*cache);
	if (!scratch)
	{
		return Failure(scratch.error());
	}
	auto response = Get(url, *scratch);
	if (!response)
	{
		return Failure(response.error());
	}
	if (response->status != 200)
	{
		fmt::print(stderr, "cmt: {} answered HTTP {}{}\n", Redacted(url), response->status,
				   response->status == 404 ? ": no file there; copy the tune's link again" : "");
		return Failure(Error::Kind::Os, "download failed");
	}
	if (share::IsSelectionOnly(response->body))
	{
		auto info = share::ValidateSelectionOnly(response->body);
		if (!info)
		{
			return Failure(info.error());
		}
	}
	else
	{
		auto info = share::Validate(response->body);
		if (!info)
		{
			fmt::print(stderr, "cmt: {} is not a valid shared tune\n", Redacted(url));
			return Failure(info.error());
		}
	}
	return Keep(*cache, *scratch, url, response->body);
}

Result<ClosestTune> DownloadClosest(std::string base, std::string_view cpu, std::string_view gpu)
{
	while (std::size(base) > 1 && base.back() == '/')
	{
		base.pop_back();
	}
	CMT_RETURN_IF_ERROR(CheckUrl(base));
	if (base.find_first_of("?#") != std::string::npos)
	{
		return UsageError(
			"--remote-config takes a server address without a query, e.g. https://cudamemcpy.com");
	}
	const auto index_url = base + "/data/summary.json";
	auto cache = CacheDirectory();
	if (!cache)
	{
		return Failure(cache.error());
	}
	auto scratch = Scratch::Create(*cache);
	if (!scratch)
	{
		return Failure(scratch.error());
	}
	auto response = Get(index_url, *scratch);
	if (!response)
	{
		return Failure(response.error());
	}
	const auto host = Redacted(base).substr(base.find("://") + 3);
	if (response->status == 404)
	{
		fmt::print(stderr, "cmt: {} has no tune for {} / {}; run: cmt tune --search quick\n", host, cpu, gpu);
		return Unsupported("no remote tune for this machine");
	}
	if (response->status != 200)
	{
		fmt::print(stderr, "cmt: {} answered HTTP {}\n", Redacted(index_url), response->status);
		return Failure(Error::Kind::Os, "download failed");
	}
	JsonParser parser;
	Json index;
	simdjson::dom::array runs;
	if (parser.parse(response->body).get(index) || index["runs"].get(runs))
	{
		return Failure(Error::Kind::Verify, "invalid static tune index");
	}
	std::string selected;
	for (Json run : runs)
	{
		std::string_view id, candidate_cpu, candidate_gpu;
		bool has_selection = false;
		if (run["id"].get(id) || run["cpu"].get(candidate_cpu) || run["gpu"].get(candidate_gpu) ||
			run["has_selection"].get(has_selection))
		{
			return Failure(Error::Kind::Verify, "invalid run in static tune index");
		}
		if (candidate_cpu == cpu && candidate_gpu == gpu && has_selection)
		{
			if (id.empty() || id == "." || id == ".." || id.size() > 128 ||
				!std::all_of(id.begin(), id.end(), [](unsigned char c)
							 { return std::isalnum(c) || c == '-' || c == '_' || c == '.'; }))
			{
				return Failure(Error::Kind::Verify, "invalid run ID in static tune index");
			}
			if (id > selected)
			{
				selected = id;
			}
		}
	}
	if (selected.empty())
	{
		fmt::print(stderr, "cmt: {} has no tune for {} / {}; run: cmt tune --search quick\n", host, cpu, gpu);
		return Unsupported("no remote tune for this machine");
	}
	const auto url = base + "/data/" + selected + "/tune.json";
	response = Get(url, *scratch);
	if (!response)
	{
		return Failure(response.error());
	}
	if (response->status != 200)
	{
		fmt::print(stderr, "cmt: {} answered HTTP {}\n", Redacted(url), response->status);
		return Failure(Error::Kind::Os, "download failed");
	}
	auto info = share::ValidateSelectionOnly(response->body);
	if (!info)
	{
		fmt::print(stderr, "cmt: {} sent no usable tune\n", host);
		return Failure(info.error());
	}
	if (info->key.arch != ArchName(NativeArch()))
	{
		fmt::print(stderr, "cmt: {}'s tune is for {}, this is {}; run: cmt tune --search quick\n", host,
				   info->key.arch, ArchName(NativeArch()));
		return Unsupported("the remote tune is for another CPU architecture");
	}
	ClosestTune tune;
	if (info->key.cpu != cpu || info->key.gpu != gpu)
	{
		return Failure(Error::Kind::Verify, "downloaded tune does not match the static index");
	}
	tune.arch = info->key.arch;
	tune.cpu = info->key.cpu;
	tune.gpu = info->key.gpu;
	auto path = Keep(*cache, *scratch, url, response->body);
	if (!path)
	{
		return Failure(path.error());
	}
	tune.path = std::move(*path);
	return tune;
}

} // namespace cmt
