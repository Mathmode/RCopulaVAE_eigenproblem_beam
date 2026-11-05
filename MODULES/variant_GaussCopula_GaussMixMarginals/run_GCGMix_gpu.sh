#!/usr/bin/env bash
# VARIANT: Gaussian copula + mixture of K truncated Gaussians per marginal, on the FINAL data set. For every K of --gmix, one
# after the other: training, post-processing of test sets A and B (same code and metrics as the proposed model), posterior
# quality (validation -ELBO, KL lower bound, Pareto-k). Runs already trained are not trained again.
#
# Options (all optional):
#   --gmix "K ..."   components per marginal (default "3 5")
#   --protocol P     conv = train to convergence, the rule of model_comparison/run_convergence_gpu.sh (default);
#                    fixed = 1500 epochs with the cosine schedule, as the Ep1500 runs of the other models
#   --gpu N          GPU index as in nvidia-smi (default 2)
#   --obs O          fv = frequencies + vertical mode shapes (default); f = frequencies only
#   --modes M        identified modes, 1-5 (default 5)
#   --seed S         random seed (default 1234)
#   --post           skip the training
# More options for the training script through the environment variable EXTRA_TRAIN_ARGS.
# Run from the project root (the folder that contains MODULES/ and Data/), with your environment active:
#     nohup bash MODULES/variant_GaussCopula_GaussMixMarginals/run_GCGMix_gpu.sh --gmix 3 --gpu 2 > GCGMix3.log 2>&1 &
set -e
GMIX="3 5"
PROTOCOL=conv
GPU=2
OBS=fv
MODES=5
SEED=1234
MODE=all
while [ $# -gt 0 ]; do
    case "$1" in
        --gmix) GMIX=$2; shift 2 ;;
        --protocol) PROTOCOL=$2; shift 2 ;;
        --gpu) GPU=$2; shift 2 ;;
        --obs) OBS=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --post) MODE=post; shift ;;
        *) echo "unknown option '$1' (options: --gmix \"K ...\" --protocol conv|fixed --gpu N --obs fv|f --modes M --seed S --post)"; exit 1 ;;
    esac
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi
DATA=Data/03Nov_FINAL_GirderBearings_7D_5sensors_5modes
TAG=03Nov-FINAL${MODES}modes          # the run name always states the number of modes and the observed quantities
case "${PROTOCOL}" in
    conv)  LAST=Conv6000; SCHEDULE=(--schedule plateau --epochs 6000) ;;
    fixed) LAST=Ep1500; SCHEDULE=(--schedule cosine --epochs 1500) ;;
    *)     echo "--protocol must be conv or fixed"; exit 1 ;;
esac

for K in ${GMIX}; do
    NAME=GCGMix${K}_${TAG}_7D_obs-${OBS}_Gamma1.0_${LAST}
    if [ "${SEED}" != "1234" ]; then NAME=${NAME}_seed${SEED}; fi
    RUN=Output/variant_GaussCopula_GaussMixMarginals/${NAME}
    echo "=== $(date) | GPU ${GPU} | ${K} truncated Gaussians per marginal | ${OBS} | ${MODES} modes | run ${RUN} ==="
    if [ "${MODE}" != "post" ]; then
        if [ -f "${RUN}/final_model_weights.weights.h5" ]; then
            echo "=== $(date) | ${RUN} already trained: training skipped ==="
        else
            echo "=== $(date) | training (${PROTOCOL}) ==="
            python MODULES/variant_GaussCopula_GaussMixMarginals/main_train_GCGMix.py --gpu "${GPU}" --gmix "${K}" \
                --data-dir "${DATA}" --n-modes "${MODES}" --obs "${OBS}" --seed "${SEED}" "${SCHEDULE[@]}" --run-name "${NAME}" ${EXTRA_TRAIN_ARGS}
        fi
    fi
    echo "=== $(date) | post-processing (test sets A and B) ==="
    python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}"
    echo "=== $(date) | posterior quality (validation -ELBO, KL lower bound, Pareto-k) ==="
    python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${RUN}" --samples 500
    echo "=== $(date) | done: ${RUN} ==="
done
