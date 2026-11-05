#!/usr/bin/env bash
# Train-to-convergence run of one posterior family on the FINAL data set, with the same rule for every family
# (review29oct_fixed/training_common.py, --schedule plateau; the training stops by the rule, not by a budget of epochs):
#   learning rate 5e-4; every 20 epochs after the warm-up the validation -ELBO is estimated with the numpy estimator of
#   evaluate_posterior_quality.py (20 samples of q per observation, same random numbers each time, Monte Carlo error about
#   0.001-0.004 nats) and the best weights are kept; the learning rate is halved after 200 epochs without an improvement
#   larger than 0.002 nats; at 1e-5, the same condition stops the training. --max-epochs is only a cap.
# Steps: training (skipped if the run already exists), post-processing of test sets A and B (all figures and metrics),
# posterior quality (validation -ELBO, KL lower bound, Pareto-k). The runs with the fixed 1500-epoch schedule are not touched.
# Afterwards, with the runs together: python MODULES/model_comparison/compare_convergence.py --obs <obs>
#
# Options:
#   --family F       ks | gctn | gmm (required)
#   --gpu N          GPU index as in nvidia-smi (default 2)
#   --obs O          fv = frequencies + vertical mode shapes (default); f = frequencies only
#   --modes M        identified modes, 1-5 (default 5)
#   --ks K           Kumaraswamy components, family ks (default 3)
#   --components C   Gaussian components, family gmm (default 5)
#   --seed S         random seed (default 1234)
#   --max-epochs E   cap of epochs (default 6000)
#   --extra "..."    more options for the training script, e.g. --extra "--lr-patience 300"
#   --post           skip the training
#   --timing         only measure the time per epoch: 60 epochs, no post-processing, written to
#                    Output/model_comparison/timing/. Use a GPU with nothing else running.
# Run from the project root (the folder that contains MODULES/ and Data/), with your environment active:
#     nohup bash MODULES/model_comparison/run_convergence_gpu.sh --family ks --ks 3 --obs f --gpu 0 > conv_KS3_f.log 2>&1 &
set -e
FAMILY=
GPU=2
OBS=fv
MODES=5
KS=3
COMP=5
SEED=1234
MAXEP=6000
EXTRA=
MODE=all
while [ $# -gt 0 ]; do
    case "$1" in
        --family) FAMILY=$2; shift 2 ;;
        --gpu) GPU=$2; shift 2 ;;
        --obs) OBS=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        --ks) KS=$2; shift 2 ;;
        --components) COMP=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --max-epochs) MAXEP=$2; shift 2 ;;
        --extra) EXTRA=$2; shift 2 ;;
        --post) MODE=post; shift ;;
        --timing) MODE=timing; shift ;;
        *) echo "unknown option '$1' (options: --family ks|gctn|gmm --gpu N --obs fv|f --modes M --ks K --components C --seed S --max-epochs E --extra \"...\" --post --timing)"; exit 1 ;;
    esac
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi
DATA=Data/03Nov_FINAL_GirderBearings_7D_5sensors_5modes
TAG=03Nov-FINAL${MODES}modes          # the run name always states the number of modes and the observed quantities
SUFFIX=obs-${OBS}_Gamma1.0_Conv${MAXEP}
if [ "${SEED}" != "1234" ]; then SUFFIX=${SUFFIX}_seed${SEED}; fi
case "${FAMILY}" in
    ks)   NAME=KSVAE_fixed_${TAG}_7D_${KS}KS_${SUFFIX}; ROOT=Output/review29oct_fixed
          TRAIN=(MODULES/review29oct_fixed/main_train_KSVAE.py --num-ks "${KS}") ;;
    gctn) NAME=GCTN_${TAG}_7D_${SUFFIX}; ROOT=Output/baseline_GaussCopula_GaussMarginals
          TRAIN=(MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCTN.py) ;;
    gmm)  NAME=GMM${COMP}full_${TAG}_7D_${SUFFIX}; ROOT=Output/baseline_GaussMixture_K5
          TRAIN=(MODULES/baseline_GaussMixture_K5/main_train_GMM.py --components "${COMP}" --covariance full --samples 2) ;;
    *)    echo "--family must be ks, gctn or gmm"; exit 1 ;;
esac
COMMON=(--gpu "${GPU}" --data-dir "${DATA}" --n-modes "${MODES}" --obs "${OBS}" --seed "${SEED}")

if [ "${MODE}" = "timing" ]; then
    echo "=== $(date) | timing ${NAME} on GPU ${GPU} ==="
    mkdir -p Output/model_comparison/timing
    python "${TRAIN[@]}" "${COMMON[@]}" --epochs 60 --run-name "$(pwd)/Output/model_comparison/timing/${NAME}"
    echo "=== $(date) | done: Output/model_comparison/timing/${NAME} ==="
    exit 0
fi

RUN=${ROOT}/${NAME}
echo "=== $(date) | GPU ${GPU} | ${FAMILY} | ${OBS} | ${MODES} modes | seed ${SEED} | run ${RUN} ==="
if [ "${MODE}" != "post" ]; then
    if [ -f "${RUN}/final_model_weights.weights.h5" ]; then
        echo "=== $(date) | ${RUN} already trained: training skipped ==="
    else
        echo "=== $(date) | training to convergence (cap ${MAXEP} epochs) ==="
        python "${TRAIN[@]}" "${COMMON[@]}" --schedule plateau --epochs "${MAXEP}" --run-name "${NAME}" ${EXTRA}
    fi
fi
echo "=== $(date) | post-processing (test sets A and B) ==="
python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}"
echo "=== $(date) | posterior quality (validation -ELBO, KL lower bound, Pareto-k) ==="
python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${RUN}" --samples 500
echo "=== $(date) | done: ${RUN} ==="
