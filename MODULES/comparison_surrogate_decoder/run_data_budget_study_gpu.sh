#!/usr/bin/env bash
# R1.4, data budget: proposed VAE (eigenvalue solver) vs VAE with the surrogate decoder of Ref. 41 when only N FE
# simulations are available for training (the surrogate is trained on the same N simulations). Noise (2.5 %), validation
# and test sets are those of the final study; only the training split is reduced (random subset, seed 2026).
#   0. reduced data set (surrogate_data_budget.py --make-data-only)
#   1. proposed VAE (eigenvalue solver)
#   2. surrogate decoder (architecture of Ref. 41, noise-free nominal targets)
#   3. VAE with the frozen surrogate
#      steps 1-3 on the SAME GPU, one after the other, with the settings of the final runs
#   4. post-processing and posterior quality of both runs (exact model)
#   5. comparison table, per-observation analysis and training times
# Options:
#   --n-train N   number of training simulations (required, e.g. 500 or 1000)
#   --gpu G       GPU index as in nvidia-smi (default 0)
# Run from the project root, with the environment active, one size per GPU:
#   nohup bash MODULES/comparison_surrogate_decoder/run_data_budget_study_gpu.sh --n-train 500 --gpu 0 > budget_500.log 2>&1 &
#   nohup bash MODULES/comparison_surrogate_decoder/run_data_budget_study_gpu.sh --n-train 1000 --gpu 1 > budget_1000.log 2>&1 &
set -e
N=""
GPU=0
while [ $# -gt 0 ]; do
    case "$1" in
        --n-train) N=$2; shift 2 ;;
        --gpu) GPU=$2; shift 2 ;;
        *) echo "unknown option '$1' (options: --n-train N --gpu G)"; exit 1 ;;
    esac
done
if [ -z "${N}" ]; then echo "--n-train is required (e.g. 500 or 1000)"; exit 1; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi

OUT=Output/comparison_surrogate_decoder
DATA=${OUT}/data_budget/data_ntrain${N}
EIGEN=${OUT}/data_budget/VAE_eigen_ntrain${N}
SUR=${OUT}/data_budget/surrogate_ntrain${N}
SURVAE=${OUT}/data_budget/VAE_surrogate_ntrain${N}
CMP=${OUT}/data_budget/comparison_ntrain${N}
mkdir -p "${CMP}"

echo "=== $(date) | ${N} training simulations | GPU ${GPU}: $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null) ==="

echo "=== $(date) | 0. reduced data set: ${DATA} ==="
python MODULES/comparison_surrogate_decoder/surrogate_data_budget.py --sizes "${N}" --make-data-only

echo "=== $(date) | 1. proposed VAE (eigenvalue solver): ${EIGEN} ==="
if [ -f "${EIGEN}/final_model_weights.weights.h5" ]; then echo "    already trained: skipped"; else
    python MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCTN.py --gpu "${GPU}" --data-dir "${DATA}" --n-modes 5 \
        --obs fv --seed 1234 --schedule plateau --epochs 6000 --run-name "$(pwd)/${EIGEN}"
fi
echo "=== $(date) | 2. surrogate decoder: ${SUR} ==="
if [ -f "${SUR}/surrogate_decoder.weights.h5" ]; then echo "    already trained: skipped"; else
    python MODULES/comparison_surrogate_decoder/train_surrogate_decoder.py --gpu "${GPU}" --data-dir "${DATA}" \
        --seed 1234 --targets nominal --hidden 10 30 50 70 80 --out-dir "${SUR}"
fi
echo "=== $(date) | 3. VAE with the frozen surrogate: ${SURVAE} ==="
if [ -f "${SURVAE}/surrogateVAE_full_model.weights.h5" ]; then echo "    already trained: skipped"; else
    python MODULES/comparison_surrogate_decoder/main_train_GCTN_surrogate.py --gpu "${GPU}" --data-dir "${DATA}" --n-modes 5 \
        --obs fv --seed 1234 --schedule plateau --epochs 6000 --surrogate-dir "${SUR}" --run-name "$(pwd)/${SURVAE}"
fi

echo "=== $(date) | training times (same GPU) ==="
python - "${EIGEN}" "${SUR}" "${SURVAE}" "${CMP}/training_times.txt" <<'EOF'
import json, os, sys
import pandas as pd
eig, sur, svae, out = sys.argv[1:5]
lines = []
for name, run in (("proposed VAE (eigenvalue solver)", eig), ("VAE with the frozen surrogate", svae)):
    h = pd.read_csv(os.path.join(run, "model_history.csv"))
    t = h["epoch_time_s"]
    lines.append(f"{name:36s} epochs {len(h):5d} | sum of epoch times {t.sum() / 60:6.1f} min | median {t.median():.3f} s/epoch "
                 f"| best val -ELBO {h['val_neg_elbo'].min():.4f}")
cfg = json.load(open(os.path.join(sur, "surrogate_config.json")))
lines.append("surrogate decoder: " + ", ".join(f"{k} = {v}" for k, v in cfg.items() if "time" in k.lower() or "epoch" in k.lower()))
text = "\n".join(lines)
print(text)
open(out, "w").write(text + "\n")
EOF

echo "=== $(date) | 4. post-processing and posterior quality (exact model) ==="
for RUN in "${EIGEN}" "${SURVAE}"; do
    python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}"
    python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${RUN}" --samples 500
done
echo "=== $(date) | 5. comparison ==="
python MODULES/comparison_surrogate_decoder/compare_eigen_vs_surrogate.py --eigen-run "${EIGEN}" --surrogate-run "${SURVAE}" \
    --out-dir "${CMP}"
python MODULES/comparison_surrogate_decoder/analyse_eigen_vs_surrogate.py --eigen-run "${EIGEN}" --surrogate-run "${SURVAE}" \
    --surrogate-dir "${SUR}" --out-dir "${CMP}/analysis"
echo "=== $(date) | done: ${CMP} ==="
