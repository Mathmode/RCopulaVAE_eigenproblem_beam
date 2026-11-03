#!/usr/bin/env bash
# Selection of the number K of Kumaraswamy components of the proposed model (FINAL data set).
# For every seed and K: training (skipped if the run already exists), whole-test-set metrics, and posterior quality on
# the VALIDATION set (-ELBO = KL to the exact posterior + const., Pareto-k).
# K = 3 with seed 1234 is the definitive run, trained by review29oct_fixed/run_FINAL_case_study.sh: leave it out of the
# K list for seed 1234 while that script runs, otherwise it would be trained twice at the same time.
# Afterwards, with all runs in one place: python MODULES/model_comparison/compare_models.py --data <data set folder>
# (selects K on the validation set only).
#
# Options (all optional):
#   --gpu N          GPU index as in nvidia-smi (default 2)
#   --seeds "S ..."  random seeds (default "1234")
#   --ks "K ..."     numbers of Kumaraswamy components (default "1 2 3 4 5")
#   --modes M        identified modes, 1-5 (default 5)
# Run from the project root, with your environment active, e.g. two seeds in parallel on the same GPU:
#     nohup bash MODULES/model_comparison/run_KS_components_gpu.sh --gpu 2 --seeds 1234 --ks "1 2 4 5" > ks_seed1234.log 2>&1 &
#     nohup bash MODULES/model_comparison/run_KS_components_gpu.sh --gpu 2 --seeds 2024 --ks "1 2 3 4 5" > ks_seed2024.log 2>&1 &
set -e
GPU=2
SEEDS=1234
KS="1 2 3 4 5"
MODES=5
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --seeds) SEEDS=$2; shift 2 ;;
        --ks) KS=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        *) echo "unknown option '$1' (options: --gpu N --seeds \"S ...\" --ks \"K ...\" --modes M)"; exit 1 ;;
    esac
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi
DATA=Data/03Nov_FINAL_GirderBearings_7D_5sensors_5modes
TAG=03Nov-FINAL${MODES}modes          # the run name always states the number of modes
echo "=== $(date) | GPU ${GPU} | seeds ${SEEDS} | K = ${KS} | ${MODES} modes ==="

for SEED in ${SEEDS}; do
    for K in ${KS}; do
        RUN=KSVAE_fixed_${TAG}_7D_${K}KS_obs-fv_Gamma1.0_Ep1500
        if [ "${SEED}" != "1234" ]; then RUN=${RUN}_seed${SEED}; fi
        if [ ! -f "Output/review29oct_fixed/${RUN}/final_model_weights.weights.h5" ]; then
            echo "=== $(date) | training K = ${K}, seed ${SEED} on GPU ${GPU} ==="
            python MODULES/review29oct_fixed/main_train_KSVAE.py --gpu "${GPU}" --epochs 1500 --data-dir "${DATA}" \
                --n-modes "${MODES}" --num-ks "${K}" --seed "${SEED}" --run-name "${RUN}"
        fi
        echo "=== $(date) | metrics and posterior quality of ${RUN} ==="
        python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}" --metrics-only
        python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${RUN}" --samples 500
    done
done
echo "=== $(date) | done (seeds ${SEEDS}, K = ${KS}, ${MODES} modes); compare with compare_models.py once all runs are together ==="
