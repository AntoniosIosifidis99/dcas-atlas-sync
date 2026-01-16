#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs logs splits

GPU="${GPU:-1}"
T="${T:-50}"
TTR_TH="${TTR_TH:-0.20}"
SEEDS=(11 12 13)

# Single setting (requested)
d="${d:-5}"
B="${B:-3}"

common_args=(
  --dataset cifar10 --model resnet18
  --mode curriculum
  --atlas_interval 2 --atlas_warmup 2
  --syn_per_class 128
  --iid 0 --classes_per_client 2
  --num_edges 2 --num_clients 10
  --num_communication "${T}" --num_local_update 1
  --batch_size 64 --lr 0.01
  --gpu "${GPU}"
)

run_if_missing () {
  local name="$1"
  local seed="$2"
  local run_dir="$3"
  local split_file="$4"
  shift 4
  local extra_args=("$@")

  if [[ -f "${run_dir}/training_metrics.csv" ]]; then
    echo "SKIP (exists): ${run_dir}"
    return 0
  fi

  echo "=============================="
  echo "RUN: ${name} | seed=${seed}"
  echo "DIR: ${run_dir}"
  echo "=============================="

  python -u hierfavg.py \
    "${common_args[@]}" \
    --seed "${seed}" \
    --split_file "${split_file}" \
    --atlas_log_dir "${run_dir}" \
    --atlas_output_dir "${run_dir}" \
    "${extra_args[@]}" \
    |& tee "logs/${name}_seed${seed}.log"
}

eval_one () {
  local run_dir="$1"
  local oracle_dir="${2:-}"
  if [[ -z "${oracle_dir}" ]]; then
    python -u eval_controlplane_metrics.py --run_dir "${run_dir}" --ttr_threshold "${TTR_TH}" --phase2 0.70
  else
    python -u eval_controlplane_metrics.py --run_dir "${run_dir}" --oracle_dir "${oracle_dir}" --ttr_threshold "${TTR_TH}" --phase2 0.70
  fi
}

for s in "${SEEDS[@]}"; do
  SPLIT="splits/cifar10_seed${s}.json"

  # Reuse existing oracle runs (do NOT rerun oracle)
  ORACLE_DIR="runs/PAPER_oracle_T${T}_seed${s}"

  if [[ ! -f "${ORACLE_DIR}/training_metrics.csv" || ! -f "${ORACLE_DIR}/cp_metrics.csv" ]]; then
    echo "ERROR: Missing oracle CSVs under ${ORACLE_DIR}"
    echo "Expected: training_metrics.csv and cp_metrics.csv"
    echo "Fix: ensure PAPER oracle runs exist for seed=${s}, T=${T}"
    exit 1
  fi

  echo "=============================="
  echo "USING ORACLE: ${ORACLE_DIR}"
  echo "=============================="

  eval_one "${ORACLE_DIR}"

  # RANDOM
  RDIR="runs/GRID_random_T${T}_seed${s}_d${d}_B${B}"
  run_if_missing "GRID_random_T${T}_d${d}_B${B}" "${s}" "${RDIR}" "${SPLIT}" \
    --cp_enable 1 --cp_delay_rounds "${d}" --cp_class_budget "${B}" \
    --cp_policy random --cp_trigger always
  eval_one "${RDIR}" "${ORACLE_DIR}"

  # ACE
  ADIR="runs/GRID_ace_T${T}_seed${s}_d${d}_B${B}"
  run_if_missing "GRID_ace_T${T}_d${d}_B${B}" "${s}" "${ADIR}" "${SPLIT}" \
    --cp_enable 1 --cp_delay_rounds "${d}" --cp_class_budget "${B}" \
    --cp_policy ace_greedy --cp_trigger always
  eval_one "${ADIR}" "${ORACLE_DIR}"

  # DCAS
  DDIR="runs/GRID_dcas_T${T}_seed${s}_d${d}_B${B}"
  run_if_missing "GRID_dcas_T${T}_d${d}_B${B}" "${s}" "${DDIR}" "${SPLIT}" \
    --cp_enable 1 --cp_delay_rounds "${d}" --cp_class_budget "${B}" \
    --cp_policy dcas_risk_greedy --cp_trigger always
  eval_one "${DDIR}" "${ORACLE_DIR}"

done

echo "DONE."
