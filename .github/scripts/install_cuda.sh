#!/usr/bin/env bash
# Install the parts of a CUDA toolkit the build reads, at pinned versions, from NVIDIA's apt repository for
# Ubuntu 22.04 x86_64: nvcc (FindCUDAToolkit requires it; no device code is compiled), the runtime and the driver
# API headers. Usage: install_cuda.sh 12.9|13.0
#
# Trust: the keyring package is checked against the SHA-256 below before dpkg installs it; apt then checks the
# repository's signed index with that key. Every CUDA package is pinned, so a build of a given commit gets the same
# toolkit. To move to a newer toolkit release, take the versions from
#   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz
# (each package's Depends line names the others) and update both lists together.
set -euo pipefail

keyring=cuda-keyring_1.1-1_all.deb
keyring_sha256=d93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df
repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64

case ${1:?usage: install_cuda.sh 12.9|13.0} in
12.9)
	packages=(
		cuda-nvcc-12-9=12.9.86-1 cuda-crt-12-9=12.9.86-1 cuda-nvvm-12-9=12.9.86-1
		cuda-cudart-12-9=12.9.79-1 cuda-cudart-dev-12-9=12.9.79-1 cuda-driver-dev-12-9=12.9.79-1
		cuda-cccl-12-9=12.9.27-1
	)
	;;
13.0)
	packages=(
		cuda-nvcc-13-0=13.0.88-1 cuda-crt-13-0=13.0.88-1 libnvvm-13-0=13.0.88-1 libnvptxcompiler-13-0=13.0.88-1
		cuda-cudart-13-0=13.0.96-1 cuda-cudart-dev-13-0=13.0.96-1 cuda-driver-dev-13-0=13.0.96-1
		cuda-cccl-13-0=13.0.85-1 cuda-culibos-dev-13-0=13.0.85-1
	)
	;;
*)
	echo "install_cuda.sh: no pinned package list for CUDA $1" >&2
	exit 2
	;;
esac

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
curl -fsSL --retry 3 -o "$work/$keyring" "$repo/$keyring"
echo "$keyring_sha256  $work/$keyring" | sha256sum --check --strict
sudo dpkg -i "$work/$keyring"
sudo apt-get update -q
sudo apt-get install -y -q --no-install-recommends "${packages[@]}"
