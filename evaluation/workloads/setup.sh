#!/usr/bin/env bash
# Prepare one training/inference workload: its Python environment and its inputs. Idempotent.
#
# Usage: setup.sh ENV [INPUT]...
#   ENV    torch251 | torch | vllm | sglang | jax | cudf     -> $CMT_ENV_ROOT/ENV (default <repo>/tmp/envs)
#   INPUT  hf:ORG/NAME   a Hugging Face model, into the normal HF cache (so models already on the machine are reused)
#          imagenette    Imagenette2-320 (13k real ImageNet JPEGs, 326 MB) -> $CMT_ENV_ROOT/data/imagenette2-320
#          pdsh-sf10     PDS-H (TPC-H) scale 10 parquet, 3.7 GB         -> $CMT_ENV_ROOT/data/pdsh-sf10
#
# Trade-offs (on purpose):
# - uv with a uv-managed CPython only. A distro or conda python may carry file capabilities; glibc then runs it
#   in secure mode and silently drops LD_PRELOAD, so neither `cmt run` nor `cmt trace` would reach the app
#   (seen with miniforge's python3.12).
# - Environments are shared where versions agree and split where they conflict: torch 2.5.1 (the checkpoint
#   reproduction), a new torch, and vLLM / SGLang / JAX / RAPIDS, which each pin their own CUDA stacks.
# - CUDA 13.0 builds (cu130) where there is a choice, so the same envs run on an R580 (CUDA 13.0) driver.
#   That is why "torch" is 2.14.1+cu130: torch 2.15 is published only as a cu132 build.
# - torch 2.5.1 (cu124) has no Blackwell (sm_120) code; its catalog entries say so in needs.max_compute.
# - vLLM (FlashInfer) and SGLang compile kernels at first use on a GPU nobody prebuilt for (the B300's
#   sm_103) and run `ninja` by name; the environment carries its own, so no system package is needed.
# - Versions are pinned. A stamp file holding the package list makes a rerun a no-op; a changed list
#   recreates the environment, so no package from an older list survives.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
root=${CMT_ENV_ROOT:-$repo/tmp/envs}
name=${1:?usage: setup.sh ENV [INPUT]...}
shift
case $name in
	torch251) pkgs="torch==2.5.1 numpy==2.1.3" ;;
	torch) pkgs="torch==2.14.1+cu130 torchvision==0.29.1+cu130 transformers==5.22.1 accelerate==1.15.0" ;;
	vllm) pkgs="vllm==0.33.0 ninja==1.13.2" ;;
	sglang) pkgs="sglang==0.5.23 ninja==1.13.2" ;;
	jax) pkgs="jax[cuda13]==0.12.0.1" ;;
	cudf) pkgs="cudf-polars-cu13==26.10.2 tpchgen-cli==4.0.0" ;;
	*) echo "setup.sh: unknown environment '$name'" >&2; exit 2 ;;
esac

env=$root/$name
if [[ "$(cat "$env/.cmt-packages" 2>/dev/null)" != "$pkgs" ]]; then
	uv venv --quiet --clear --python 3.12 --python-preference only-managed "$env"
	# shellcheck disable=SC2086 # pkgs is a word list on purpose
	VIRTUAL_ENV=$env uv pip install --quiet --index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cu130 $pkgs
	echo "$pkgs" >"$env/.cmt-packages"
fi

for input in "$@"; do
	case $input in
	hf:*)
		# `hf download` checks the cache first, so this is cheap when the model is already there.
		uvx --quiet --from 'huggingface_hub[cli]' hf download --quiet "${input#hf:}" >/dev/null
		;;
	imagenette)
		dest=$root/data/imagenette2-320
		if [[ ! -d $dest/val ]]; then
			mkdir -p "$root/data"
			# fast.ai's archive (2021), checked against its SHA-256 before anything is unpacked.
			archive=$root/data/imagenette2-320.tgz.partial
			curl -fsSL -o "$archive" https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
			if ! echo "569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c  $archive" |
				sha256sum --check --strict --quiet; then
				rm -f "$archive"
				echo "setup.sh: imagenette2-320.tgz does not have the expected SHA-256" >&2
				exit 1
			fi
			tar -xzf "$archive" -C "$root/data"
			rm "$archive"
		fi
		;;
	pdsh-sf10)
		dest=$root/data/pdsh-sf10
		# Optional cache: the same SF10 tables, if present in this repository's tmp/ (tpchgen output is deterministic).
		cached=$repo/tmp/workloads/data/pdsh/sf10
		if [[ ! -e $dest/lineitem.parquet ]]; then
			mkdir -p "$root/data"
			if [[ -e $cached/lineitem.parquet ]]; then
				ln -sfn "$cached" "$dest"
			else
				"$root/cudf/bin/tpchgen-cli" parquet -s 10 --output-dir "$dest.partial"
				mv "$dest.partial" "$dest"
			fi
		fi
		;;
	*) echo "setup.sh: unknown input '$input'" >&2; exit 2 ;;
	esac
done
echo "setup.sh: $name ready at $env"
