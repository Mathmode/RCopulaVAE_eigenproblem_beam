#!/usr/bin/env bash
# R1.7 / R3.8: controlled timing of the training. The three setups are trained one after the other, in the same session,
# on the same GPU, with the same code and settings as the final runs (batch 256, H = 6, plateau schedule with the
# validation monitor), for a fixed number of epochs. Only the data set (and so D) changes:
#   D7_final   Data/13Nov_FINAL_SlabS10_7D_DoE30           (final model of the revised manuscript, D = 7)
#   D7_06Nov   Data/06Nov_FINAL_SlabS10_7D_5sensors_5modes  (D = 7 run of the scalability table of R1.7)
#   D12_11Nov  Data/11Nov_FINAL_SlabS10_12D_5sensors_5modes (D = 12 run of the scalability table of R1.7)
# Run it when the server is otherwise idle (no other training or post-processing jobs: the eigenvalue problems run on
# the CPU). The log records the hardware and the load before each setup.
# Usage (project root, environment active):
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_timing_R38_gpu.sh --gpu 2 --epochs 100 > timing_R38.log 2>&1 &
set -e
GPU=2
EPOCHS=100
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --epochs) EPOCHS=$2; shift 2 ;;
        *) echo "unknown option '$1' (options: --gpu N --epochs N)"; exit 1 ;;
    esac
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi
OUT=Output/timing_R38
mkdir -p "${OUT}"

echo "=== $(date) | hardware ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Socket\(s\)"
free -g | head -2
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"

for SETUP in D7_final D7_06Nov D12_11Nov; do
    case "${SETUP}" in
        D7_final)  DATA=Data/13Nov_FINAL_SlabS10_7D_DoE30 ;;
        D7_06Nov)  DATA=Data/06Nov_FINAL_SlabS10_7D_5sensors_5modes ;;
        D12_11Nov) DATA=Data/11Nov_FINAL_SlabS10_12D_5sensors_5modes ;;
    esac
    RUN="$(pwd)/${OUT}/${SETUP}"
    if [ -f "${RUN}/model_history.csv" ]; then echo "=== ${SETUP}: already timed, skipped"; continue; fi
    [ -f "${DATA}/problem_config.json" ] || { echo "=== ${SETUP}: ${DATA} not found, skipped"; continue; }
    echo "=== $(date) | ${SETUP} | ${DATA} | GPU ${GPU} ==="
    echo "    load average before: $(cut -d' ' -f1-3 /proc/loadavg)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    python MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCLogN.py --data-dir "${DATA}" --gpu "${GPU}" \
        --obs fv --seed 1234 --schedule plateau --epochs "${EPOCHS}" --run-name "${RUN}"
    echo "    load average after: $(cut -d' ' -f1-3 /proc/loadavg)"
done

echo "=== $(date) | summary (first epoch excluded: graph tracing) ==="
python - <<'PY'
import os, json
import numpy as np, pandas as pd
out = "Output/timing_R38"
for s in ("D7_final", "D7_06Nov", "D12_11Nov"):
    h = os.path.join(out, s, "model_history.csv")
    if not os.path.exists(h):
        print(f"{s:10s} missing"); continue
    t = pd.read_csv(h)["epoch_time_s"].to_numpy()[1:]
    cfg = json.load(open(os.path.join(out, s, "run_config.json")))
    p = np.load(os.path.join(out, s, "Problem_info.npy"), allow_pickle=True).item()
    print(f"{s:10s} D={p['n_dims']:2d} params={cfg['trainable_parameters']} epochs={len(t) + 1} "
          f"median {np.median(t):.3f} s/epoch (p5 {np.percentile(t, 5):.3f}, p95 {np.percentile(t, 95):.3f}) "
          f"monitor {p['monitor_time_sec']:.1f} s")
PY
echo "=== $(date) | done ==="
