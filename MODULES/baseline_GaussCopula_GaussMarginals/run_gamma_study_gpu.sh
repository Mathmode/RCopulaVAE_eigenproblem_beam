#!/usr/bin/env bash
# Sensitivity analysis to the entropy weight gamma (R3.2) for the Gaussian-copula + truncated-Gaussian VAE
# (MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCTN.py).
#
# Loss:  tau * E_q[NLL(z; m)] - gamma^2 * H[q(z | m)]   (tau -> 1 after warm-up).
#   gamma = 1.0  -> exact negative ELBO (targets the Bayesian posterior p(z | m) proportional to exp(-NLL) p(z)).
#   gamma != 1.0 -> targets the tempered posterior p_gamma(z | m) proportional to exp(-NLL / gamma^2) p(z).
#
# Because gamma^2 weights the entropy gradient during encoder optimization, each value of gamma requires a separate
# training run (all other settings identical: same data, likelihood, warm-up, seed, and plateau convergence rule).
# The gamma = 1.0 run is automatically skipped if it has already been trained.
#
# Steps for each gamma:
#   1. Train GC-TN to convergence (--schedule plateau, cap 6000 epochs; skipped if weights already exist)
#   2. Post-processing on test sets A and B + SMC reference posteriors at the same gamma (skipped if already done)
#   3. Posterior quality on validation, test A and test B (ELBO, KL lower bound, Pareto-k, validation calibration)
#   4. Summary tables and figures across all available gamma runs (gamma_sensitivity_study.py)
#
# Options:
#   --gpu N             GPU index as in nvidia-smi (default 0)
#   --gammas "G ..."    space-separated list of gamma values (default "0.5 0.75 1.0 1.5 2.0")
#   --obs O             observed quantities: fv (default) or f
#   --modes M           identified modes, 1-5 (default 5)
#   --seed S            random seed (default 1234)
#   --max-epochs E      epoch cap for the convergence controller (default 6000)
#   --metrics-only      in post-processing, skip the SMC reference posteriors and corner plots (much faster on CPU)
#   --post              skip training; only run post-processing, posterior quality and the comparison
#   --compare           only run step 4 (gamma_sensitivity_study.py) on the already-evaluated runs
#
# Examples (run from the project root, with your conda environment active):
#   # A) Run all gammas sequentially on one GPU (e.g. GPU 0):
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_gamma_study_gpu.sh --gpu 0 > gamma_study.log 2>&1 &
#
#   # B) Or parallelize across multiple GPUs (one gamma per GPU; gamma=1.0 is already trained):
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_gamma_study_gpu.sh --gpu 0 --gammas "0.5"  > gamma_0p5.log  2>&1 &
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_gamma_study_gpu.sh --gpu 1 --gammas "0.75" > gamma_0p75.log 2>&1 &
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_gamma_study_gpu.sh --gpu 2 --gammas "1.5"  > gamma_1p5.log  2>&1 &
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_gamma_study_gpu.sh --gpu 3 --gammas "2.0"  > gamma_2p0.log  2>&1 &
set -e

GPU=0
GAMMAS="0.5 0.75 1.0 1.5 2.0"
OBS=fv
MODES=5
SEED=1234
MAXEP=6000
POST_FLAGS=""
MODE=all

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --gammas) GAMMAS=$2; shift 2 ;;
        --obs) OBS=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --max-epochs) MAXEP=$2; shift 2 ;;
        --metrics-only) POST_FLAGS="--metrics-only"; shift ;;
        --post) MODE=post; shift ;;
        --compare) MODE=compare; shift ;;
        *) echo "unknown option '$1' (options: --gpu N --gammas \"G ...\" --obs fv|f --modes M --seed S --max-epochs E --metrics-only --post --compare)"; exit 1 ;;
    esac
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi

DATA=Data/06Nov_FINAL_SlabS10_7D_5sensors_5modes
ROOT=Output/baseline_GaussCopula_GaussMarginals

if [ "${MODE}" != "compare" ]; then
    for G in ${GAMMAS}; do
        # Normalize gamma string (e.g. 1 -> 1.0) as Python float formatting does
        G_FMT=$(python -c "print(float('${G}'))")
        # Short folder name (stays well below the 260-char Windows path limit when copied to PC)
        SHORT_NAME=GCTN_06Nov-${MODES}modes_7D_Gamma${G_FMT}_Conv${MAXEP}
        # Long folder name used by run_convergence_gpu.sh (checked so an existing gamma=1.0 run is reused)
        LONG_NAME=GCTN_06Nov-FINAL${MODES}modes_7D_obs-${OBS}_Gamma${G_FMT}_Conv${MAXEP}
        if [ "${OBS}" != "fv" ]; then SHORT_NAME=GCTN_06Nov-${MODES}modes_7D_obs-${OBS}_Gamma${G_FMT}_Conv${MAXEP}; fi
        if [ "${SEED}" != "1234" ]; then
            SHORT_NAME=${SHORT_NAME}_seed${SEED}
            LONG_NAME=${LONG_NAME}_seed${SEED}
        fi

        if [ -f "${ROOT}/${LONG_NAME}/final_model_weights.weights.h5" ] && [ ! -f "${ROOT}/${SHORT_NAME}/final_model_weights.weights.h5" ]; then
            NAME=${LONG_NAME}
        else
            NAME=${SHORT_NAME}
        fi
        RUN=${ROOT}/${NAME}

        echo "======================================================================"
        echo "=== $(date) | GPU ${GPU} | gamma = ${G_FMT} | ${OBS} | ${MODES} modes | run ${RUN} ==="
        echo "======================================================================"

        if [ "${MODE}" != "post" ]; then
            if [ -f "${RUN}/final_model_weights.weights.h5" ]; then
                echo "=== $(date) | ${RUN} already trained: training skipped ==="
            else
                echo "=== $(date) | 1. Training GC-TN with gamma = ${G_FMT} to convergence (cap ${MAXEP} epochs) ==="
                python MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCTN.py \
                    --gpu "${GPU}" --data-dir "${DATA}" --n-modes "${MODES}" --obs "${OBS}" \
                    --seed "${SEED}" --gamma "${G_FMT}" --schedule plateau --epochs "${MAXEP}" \
                    --run-name "${NAME}"
            fi
        fi

        if [ -f "${RUN}/POSTPROCESSING_DONE.txt" ]; then
            echo "=== $(date) | 2. Post-processing already complete for ${RUN}: skipped ==="
        else
            echo "=== $(date) | 2. Post-processing (test sets A and B) for ${RUN} ==="
            python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}" ${POST_FLAGS}
        fi

        if [ -f "${RUN}/Posterior_quality/ELBO_summary.json" ] && [ -f "${RUN}/Posterior_quality/Validation_set/Manuscript_Metrics.json" ]; then
            echo "=== $(date) | 3. Posterior quality already evaluated for ${RUN}: skipped ==="
        else
            echo "=== $(date) | 3. Posterior quality (validation, test A, test B) for ${RUN} ==="
            python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${RUN}" --samples 500
        fi
    done
fi

echo "=== $(date) | 4. Running gamma sensitivity comparison and plotting ==="
python MODULES/baseline_GaussCopula_GaussMarginals/gamma_sensitivity_study.py \
    --modes "${MODES}" --obs "${OBS}" --seed "${SEED}"
echo "=== $(date) | done ==="
