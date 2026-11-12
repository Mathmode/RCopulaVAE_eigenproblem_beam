#!/usr/bin/env bash
# R1.7, scalability of the latent space: the proposed GC-LogN model with 10 deck zones of 1 m + 2 bearings = 12 latent
# variables (instead of 5 zones of 2 m + 2 bearings = 7), with the SAME structure, FE models (10-element decoder,
# 50-element test model), 5 sensors, 5 modes, noise, modelling error and training settings as the 7D reference run
# GCLogN_06Nov-5modes_7D_Gamma1.0_Conv6000. Everything is switched by CVAE_ZONES=10 (config.py).
#   1. data set Data/11Nov_FINAL_SlabS10_12D_5sensors_5modes (skipped if it exists)
#   2. training                      -> Output/baseline_GaussCopula_GaussMarginals/GCLogN_12D
#   3. post-processing (benchmarks with --single-corners, violins) and posterior quality, as the 7D run
#   4. cost summary: GCLogN_12D/cost_summary.txt (training time, epochs, s/epoch, parameters)
# Run from the project root, with the environment active:
#   nohup bash MODULES/review29oct_fixed/run_12D_gpu.sh --gpu 0 > GCLogN_12D.log 2>&1 &
set -e
GPU=0
MAXEP=6000
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --max-epochs) MAXEP=$2; shift 2 ;;
        *) echo "unknown option '$1' (options: --gpu N --max-epochs E)"; exit 1 ;;
    esac
done
export CVAE_ZONES=10
export PYTHONUNBUFFERED=1          # live log under nohup
# small (22-DOF) eigenproblems: many BLAS threads only compete with the other jobs of the server
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi
DATA=Data/11Nov_FINAL_SlabS10_12D_5sensors_5modes
RUN=Output/baseline_GaussCopula_GaussMarginals/GCLogN_12D

echo "=== $(date) | GPU ${GPU}: $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null) | 12 latent variables ==="
echo "=== $(date) | 1. data set ${DATA} ==="
if [ -f "${DATA}/problem_config.json" ]; then echo "    exists: skipped"; else
    python MODULES/review29oct_fixed/dataset_generation.py --data-dir "${DATA}"
fi
echo "=== $(date) | 2. training (cap ${MAXEP} epochs) -> ${RUN} ==="
if [ -f "${RUN}/final_model_weights.weights.h5" ]; then echo "    already trained: skipped"; else
    python MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCLogN.py --gpu "${GPU}" --data-dir "${DATA}" --n-modes 5 \
        --obs fv --seed 1234 --schedule plateau --epochs "${MAXEP}" --run-name "$(pwd)/${RUN}"
fi
echo "=== $(date) | 3. post-processing and posterior quality ==="
python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}" --single-corners
python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${RUN}" --samples 500
echo "=== $(date) | 4. cost summary ==="
python - "${RUN}" <<'EOF'
import json, os, sys
import numpy as np, pandas as pd
run = sys.argv[1]
info = np.load(os.path.join(run, "Problem_info.npy"), allow_pickle=True).item()
cfg = json.load(open(os.path.join(run, "run_config.json")))
h = pd.read_csv(os.path.join(run, "model_history.csv"))
D = int(info["n_dims"])
lines = [f"latent variables D = {D}; posterior parameters per observation = 2D + D(D-1)/2 = {2 * D + D * (D - 1) // 2}",
         f"trainable parameters (encoder) = {cfg.get('trainable_parameters')}",
         f"epochs = {len(h)}; training time = {float(info.get('training_time_sec', np.nan)) / 60:.1f} min; "
         f"median time per epoch = {h['epoch_time_s'].median():.3f} s; best epoch = {info.get('best_epoch')}"]
open(os.path.join(run, "cost_summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
echo "=== $(date) | done: ${RUN} ==="
