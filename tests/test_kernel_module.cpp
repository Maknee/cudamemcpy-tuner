#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <unistd.h>

#include "common/error.h"
#include "tests/check.h"
#include "machine.h"

namespace {
struct Tree
{
	std::string root;

	static cmt::Result<Tree> Create()
	{
		Tree tree;
		char name[] = "/tmp/cmt-module-test-XXXXXX";
		const char* directory = mkdtemp(name);
		if (!directory)
		{
			return cmt::Failure(cmt::Error::Kind::Os, "mkdtemp failed", errno);
		}
		tree.root = directory;
		CMT_RETURN_IF_ERROR(tree.Write("/sys/module/nvidia/srcversion", "ABC123\n"));
		CMT_RETURN_IF_ERROR(tree.Write("/sys/module/nvidia_uvm/srcversion", "DEF456\n"));
		CMT_RETURN_IF_ERROR(
			tree.Write("/proc/driver/nvidia/version",
					   "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 615.71.09\n"));
		return tree;
	}

	Tree(Tree&& other) noexcept : root(std::exchange(other.root, {}))
	{
	}

	Tree(const Tree&) = delete;
	Tree& operator=(const Tree&) = delete;

	~Tree()
	{
		if (!root.empty())
		{
			std::error_code error;
			std::filesystem::remove_all(root, error);
		}
	}

	cmt::Result<void> Write(const std::string& path, const std::string& text)
	{
		std::error_code error;
		std::filesystem::create_directories(std::filesystem::path(root + path).parent_path(), error);
		if (error)
		{
			return cmt::Failure(cmt::Error::Kind::Os, "cannot create fixture directories", error.value());
		}
		std::ofstream file(root + path);
		file << text;
		file.close();
		if (!file)
		{
			return cmt::Failure(cmt::Error::Kind::Os, "cannot write fixture");
		}
		return {};
	}

	cmt::nv::KernelPatchStatus Inspect() const
	{
		return cmt::nv::InspectKernelPatches(root, root + "/sets");
	}

	cmt::Result<void> Set(const std::string& name, const std::string& uvm, const std::string& patches)
	{
		CMT_RETURN_IF_ERROR(
			Write("/sets/" + name + "/SRCVERSIONS", "nvidia.ko ABC123\nnvidia-uvm.ko " + uvm + "\n"));
		return Write("/sets/" + name + "/PATCHES", patches + "\n");
	}

private:
	Tree() = default;
};
} // namespace

TEST(version_and_open_banner_do_not_prove_a_patch)
{
	auto created = Tree::Create();
	CHECK(created);
	if (!created)
	{
		std::fprintf(stderr, "%s\n", created.error().Message());
		return;
	}
	auto tree = std::move(*created);
	const auto result = tree.Inspect();
	CHECK(result.open_kernel == true);
	CHECK(!result.patched.has_value());
}

TEST(both_modules_must_match)
{
	auto created = Tree::Create();
	CHECK(created);
	if (!created)
	{
		std::fprintf(stderr, "%s\n", created.error().Message());
		return;
	}
	auto tree = std::move(*created);
	CHECK(tree.Set("patched", "BAD", "a b"));
	CHECK(!tree.Inspect().patched.has_value());
	CHECK(tree.Set("patched", "DEF456", "a b"));
	CHECK(tree.Inspect().patched == true);
	CHECK(tree.Inspect().set == "patched");
}

TEST(stock_unknown_and_ambiguous_are_distinct)
{
	auto created = Tree::Create();
	CHECK(created);
	if (!created)
	{
		std::fprintf(stderr, "%s\n", created.error().Message());
		return;
	}
	auto tree = std::move(*created);
	CHECK(tree.Set("stock", "DEF456", "none"));
	CHECK(tree.Inspect().patched == false);
	CHECK(tree.Set("duplicate", "DEF456", "a"));
	CHECK(!tree.Inspect().patched.has_value());
	CHECK_EQ(std::size(tree.Inspect().matching_sets), std::size_t{2});
}

TEST(missing_description_and_duplicate_entries_cannot_claim_a_patch)
{
	auto created = Tree::Create();
	CHECK(created);
	if (!created)
	{
		std::fprintf(stderr, "%s\n", created.error().Message());
		return;
	}
	auto tree = std::move(*created);
	CHECK(tree.Set("patched", "DEF456", "a"));
	std::error_code error;
	CHECK(std::filesystem::remove(tree.root + "/sets/patched/PATCHES", error));
	CHECK(!error);
	if (error)
	{
		return;
	}
	CHECK(!tree.Inspect().patched.has_value());
	CHECK(tree.Write("/sets/patched/PATCHES", "a\n"));
	CHECK(tree.Write("/sets/patched/SRCVERSIONS",
					 "nvidia.ko ABC123\nnvidia.ko ABC123\nnvidia-uvm.ko DEF456\n"));
	CHECK(!tree.Inspect().patched.has_value());
}

TEST_MAIN()
