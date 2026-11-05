#!/usr/bin/env bash
# BASELINE 1 (R1.4; Ref. 41 / submitted version): Gaussian copula + one truncated Gaussian per marginal, trained with
# the ELBO, likelihood and schedule of the proposed method on the FINAL data set.
# Steps: training on one GPU, post-processing of test sets A and B (same code and metrics as the proposed model),
# posterior quality (validation -ELBO, Pareto-k) for the comparison.
#
# Options (all optional):
#   --gpu N     GPU index as in nvidia-smi (default 2)
#   --modes M   identified modes, 1-5 (default 5); M < 5 uses the first M modes of the same observations
#   --post      skip the training
# Run from the project root (the folder that contains MODULES/ and Data/), with your environment active:
#     nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_GCTN_gpu.sh --gpu 2 --modes 5 > baseline_GCTN.log 2>&1 &
set -e
GPU=2
MODES=5
MODE=all
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        --post) MODE=post; shift ;;
        *) echo "unknown option '$1' (options: --gpu N --modes M --post)"; exit 1 ;;
    esac
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi
DATA=Data/03Nov_FINAL_GirderBearings_7D_5sensors_5modes
TAG=03Nov-FINAL${MODES}modes          # the run name always states the number of modes
NAME=GCTN_${TAG}_7D_obs-fv_Gamma1.0_Ep1500
RUN=Output/baseline_GaussCopula_GaussMarginals/${NAME}
echo "=== $(date) | GPU ${GPU} | ${MODES} modes | run ${RUN} ==="

if [ "${MODE}" != "post" ]; then
    echo "=== $(date) | training baseline GC-TN ==="
    python MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCTN.py --gpu "${GPU}" --epochs 1500 \
        --data-dir "${DATA}" --n-modes "${MODES}" --run-name "${NAME}"
fi
echo "=== $(date) | post-processing (test sets A and B) ==="
python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}"
echo "=== $(date) | posterior quality (validation -ELBO, Pareto-k) ==="
python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${RUN}" --samples 500
echo "=== $(date) | done: ${RUN} ==="
