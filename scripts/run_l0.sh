#!/usr/bin/env bash
set -euo pipefail
set +u
source ~/.bashrc
set -u
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -r "$PROJECT_ROOT/../storage_env.sh" ]]; then
  source "$PROJECT_ROOT/../storage_env.sh"
fi
: "${WORK:?Set WORK to the verified data-disk work directory}"
: "${STALEMOM_OUT:?Set STALEMOM_OUT on the data disk}"
: "${STALEMOM_CKPT:?Set STALEMOM_CKPT on the data disk}"
source "${CONDA_BASE:-$(conda info --base)}/etc/profile.d/conda.sh"
conda activate "$WORK/envs/stalemom"
cd "$WORK/stalemom"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MAX_JOBS=2
export TOKENIZERS_PARALLELISM=false
mkdir -p "$STALEMOM_OUT/e5/l0"
python -m llm.run_l0 --config llm/configs/l0/smoke.yaml "$@" 2>&1 | tee "$STALEMOM_OUT/e5/l0/run_l0.out"
