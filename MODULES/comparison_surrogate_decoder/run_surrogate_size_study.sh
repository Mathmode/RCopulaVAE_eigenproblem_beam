#!/usr/bin/env bash
# Network-size study of the surrogate decoder (nominal targets, convergence rule of train_surrogate_decoder.py). The networks
# are trained one after the other on the same GPU, so their training times are comparable. Results:
# Output/comparison_surrogate_decoder/size_study/size_study.csv
# Run from the project root:  nohup bash MODULES/comparison_surrogate_decoder/run_surrogate_size_study.sh --gpu 0 > size_study.log 2>&1 &
set -e
GPU=0
MODES=5
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        *) echo "unknown option '$1' (options: --gpu N --modes M)"; exit 1 ;;
    esac
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
DATA=Data/06Nov_FINAL_SlabS10_7D_5sensors_5modes
OUT=Output/comparison_surrogate_decoder/size_study
mkdir -p "${OUT}"
for H in "10 30 50 70 80" "32 32" "64 64" "64 64 64" "128 128"; do
    N=H${H// /-}
    if [ -f "${OUT}/${N}/surrogate_config.json" ]; then echo "${N}: done before, skipped"; continue; fi
    echo "=== $(date) | ${N} ==="
    python MODULES/comparison_surrogate_decoder/train_surrogate_decoder.py --gpu "${GPU}" --data-dir "${DATA}" \
        --n-modes "${MODES}" --targets nominal --hidden ${H} --out-dir "${OUT}/${N}" > "${OUT}/${N}.log" 2>&1
done
python - <<'PY'
import json, os
import pandas as pd
OUT = "Output/comparison_surrogate_decoder/size_study"
rows = []
for n in sorted(d for d in os.listdir(OUT) if os.path.isfile(os.path.join(OUT, d, "surrogate_config.json"))):
    c = json.load(open(os.path.join(OUT, n, "surrogate_config.json")))
    a = pd.read_csv(os.path.join(OUT, n, "surrogate_accuracy.csv"))
    v, u = a[a.set == "val"], a[a.set == "uniform prior box"]
    rows.append({"hidden": n, "parameters": c["trainable_parameters"], "epochs": c["epochs_run"], "converged": c["converged"],
                 "time_min": round(c["training_time_sec"] / 60, 2), "s_per_epoch": round(c["time_per_epoch_sec"], 3),
                 "val_mean_abs_freq_err_pct": round(v.freq_err_mean_abs_pct.mean(), 3),
                 "val_max_abs_freq_err_pct": round(v.freq_err_max_abs_pct.max(), 2),
                 "val_mean_1_minus_MAC": float(v.one_minus_MAC_mean.mean()),
                 "box_mean_abs_freq_err_pct": round(u.freq_err_mean_abs_pct.mean(), 3),
                 "box_max_abs_freq_err_pct": round(u.freq_err_max_abs_pct.max(), 2)})
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "size_study.csv"), index=False)
pd.set_option("display.width", 250)
print(t.to_string(index=False))
PY
echo "=== $(date) | done: ${OUT}/size_study.csv ==="
