#!/usr/bin/env bash
# R1.4, noise-precision study: does the exact forward operator (eigenvalue solver) give better posteriors than the surrogate
# decoder of Ref. 41 when the frequencies are identified more precisely? For ONE level of the identification noise of the
# frequencies (the mode-shape noise stays at 5 % of the peak), this script
#   0. generates the data set: identical to the final one except for that noise level (same seeds; likelihood calibrated
#      again with the new noise),
#   1. trains the proposed VAE (eigenvalue solver),
#   2. trains the surrogate decoder (architecture of Ref. 41, noise-free nominal targets) and
#   3. trains the VAE with the frozen surrogate,
#      steps 1-3 on the SAME GPU, one after the other, with the settings of the final runs (so the training times can be
#      compared),
#   4. post-processing and posterior quality of both runs, with the exact model,
#   5. the paired comparison (comparison table, per-observation -ELBO and KL lower bound) and the training times.
# Options:
#   --freq-noise P   identification noise of the frequencies in % (e.g. 1 or 0.5). P = 2.5 is the final data set: it is not
#                    regenerated, the runs get the suffix _sameGPU and only steps 1-3 and the times are done (fair timing)
#   --gpu N          GPU index as in nvidia-smi (default 0)
# Run from the project root, with your environment active (one noise level per GPU), e.g.:
#   nohup bash MODULES/comparison_surrogate_decoder/run_noise_precision_study_gpu.sh --freq-noise 1 --gpu 0 > noise_1pct.log 2>&1 &
#   nohup bash MODULES/comparison_surrogate_decoder/run_noise_precision_study_gpu.sh --freq-noise 0.5 --gpu 1 > noise_0p5pct.log 2>&1 &
set -e
P=""
GPU=0
while [ $# -gt 0 ]; do
    case "$1" in
        --freq-noise) P=$2; shift 2 ;;
        --gpu) GPU=$2; shift 2 ;;
        *) echo "unknown option '$1' (options: --freq-noise P --gpu N)"; exit 1 ;;
    esac
done
if [ -z "${P}" ]; then echo "--freq-noise is required (in %, e.g. 1 or 0.5)"; exit 1; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi

PT=${P//./p}
FINAL=Data/06Nov_FINAL_SlabS10_7D_5sensors_5modes
OUT=Output/comparison_surrogate_decoder
if [ "${P}" = "2.5" ]; then
    DATA=${FINAL}; NB=${OUT}/timing_sameGPU
else
    DATA=${FINAL}_fnoise${PT}pct; NB=${OUT}/noise_${PT}pct
fi
EIGEN=${NB}/VAE_eigen
SUR=${NB}/surrogate
SURVAE=${NB}/VAE_surrogate
CMP=${NB}/comparison
mkdir -p "${CMP}"

echo "=== $(date) | frequency noise ${P} % | GPU ${GPU}: $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null) ==="
nvidia-smi -i "${GPU}" --query-gpu=utilization.gpu,memory.used --format=csv 2>/dev/null || true

if [ "${P}" != "2.5" ] && [ ! -f "${DATA}/problem_config.json" ]; then
    echo "=== $(date) | 0. data set ${DATA} ==="
    python MODULES/comparison_surrogate_decoder/make_noise_dataset.py --freq-noise "$(python -c "print(${P} / 100)")" \
        --data-dir "${DATA}" --version "06Nov-Fnoise${PT}pct_MolinerS10_kappa005_5sensors_5modes_mass5_localizedTestB"
fi

echo "=== $(date) | 1. proposed VAE (eigenvalue solver): ${EIGEN} ==="
if [ -f "${EIGEN}/final_model_weights.weights.h5" ]; then echo "    already trained: skipped"; else
    python MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCTN.py --gpu "${GPU}" --data-dir "${DATA}" --n-modes 5 \
        --obs fv --seed 1234 --schedule plateau --epochs 6000 --run-name "$(pwd)/${EIGEN}"
fi
echo "=== $(date) | 2. surrogate decoder: ${SUR} ==="
if [ -f "${SUR}/surrogate_decoder.weights.h5" ]; then echo "    already trained: skipped"; else
    python MODULES/comparison_surrogate_decoder/train_surrogate_decoder.py --gpu "${GPU}" --data-dir "${DATA}" --n-modes 5 \
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

if [ "${P}" = "2.5" ]; then echo "=== $(date) | done (timing only): ${CMP}/training_times.txt ==="; exit 0; fi

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
