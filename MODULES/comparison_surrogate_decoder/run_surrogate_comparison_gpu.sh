#!/usr/bin/env bash
# Comparison with the Gaussian copula VAE of Ref. 41 (R1.4): the proposed model uses the differentiable eigenvalue solver as
# decoder; the approach of Ref. 41 uses a surrogate neural network, pre-trained at the damage states of the same training
# data and then frozen.
#   1. pre-train the surrogate decoder (skipped if it exists)
#   2. train the VAE with the frozen surrogate (train-to-convergence rule of the proposed model; skipped if it exists)
#   3. post-processing (test sets A and B) and posterior quality, both with the exact model
#   4. comparison with the proposed run, if it exists (otherwise: run it with run_convergence_gpu.sh --family gctn, then --compare)
# Options:
#   --gpu N         GPU index as in nvidia-smi (default 0)
#   --obs O         fv (default) or f
#   --modes M       identified modes (default 5)
#   --seed S        seed of the VAE (default 1234)
#   --max-epochs E  cap of the VAE training (default 6000)
#   --targets T     targets of the surrogate at the training damage states: nominal (default; noise-free modal data of
#                   the nominal model, as when a surrogate is trained on simulations) or observed (the noisy observations)
#   --hidden "H .." hidden layers of the surrogate (default "10 30 50 70 80", the architecture of Ref. 41)
#   --compare       only step 4
# Run from the project root, with your environment active:
#   nohup bash MODULES/comparison_surrogate_decoder/run_surrogate_comparison_gpu.sh --gpu 0 > surrogate_comparison.log 2>&1 &
set -e
GPU=0
OBS=fv
MODES=5
SEED=1234
MAXEP=6000
TARGETS=nominal
HIDDEN="10 30 50 70 80"
MODE=all
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --obs) OBS=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --max-epochs) MAXEP=$2; shift 2 ;;
        --targets) TARGETS=$2; shift 2 ;;
        --hidden) HIDDEN=$2; shift 2 ;;
        --compare) MODE=compare; shift ;;
        *) echo "unknown option '$1' (options: --gpu N --obs fv|f --modes M --seed S --max-epochs E --targets nominal|observed --hidden \"H ..\" --compare)"; exit 1 ;;
    esac
done
case "${TARGETS}" in nominal|observed) ;; *) echo "--targets must be nominal or observed"; exit 1 ;; esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi
DATA=Data/06Nov_FINAL_SlabS10_7D_5sensors_5modes
TAG=06Nov-FINAL${MODES}modes
OUT=Output/comparison_surrogate_decoder
HTAG=H${HIDDEN// /-}
SUFFIX=obs-${OBS}_Gamma1.0_Conv${MAXEP}
if [ "${SEED}" != "1234" ]; then SUFFIX=${SUFFIX}_seed${SEED}; fi
# short folder names (Windows paths must stay below 260 characters): only the options that differ from the
# default configuration are appended
VAR=""
if [ "${TARGETS}" != "nominal" ]; then VAR=${VAR}_${TARGETS}; fi
if [ "${HIDDEN}" != "10 30 50 70 80" ]; then VAR=${VAR}_${HTAG}; fi
if [ "${MODES}" != "5" ]; then VAR=${VAR}_${MODES}modes; fi
if [ "${OBS}" != "fv" ]; then VAR=${VAR}_obs-${OBS}; fi
if [ "${SEED}" != "1234" ]; then VAR=${VAR}_seed${SEED}; fi
if [ "${MAXEP}" != "6000" ]; then VAR=${VAR}_ep${MAXEP}; fi
SUR=${OUT}/surrogate${VAR}
RUN=${OUT}/VAE_surrogate${VAR}
EIGEN=Output/baseline_GaussCopula_GaussMarginals/GCTN_${TAG}_7D_${SUFFIX}

if [ "${MODE}" = "all" ]; then
    echo "=== $(date) | 1. surrogate decoder, targets ${TARGETS} (${SUR}) ==="
    if [ -f "${SUR}/surrogate_decoder.weights.h5" ]; then
        echo "    already trained: skipped"
    else
        python MODULES/comparison_surrogate_decoder/train_surrogate_decoder.py --gpu "${GPU}" --data-dir "${DATA}" \
            --n-modes "${MODES}" --seed "${SEED}" --targets "${TARGETS}" --hidden ${HIDDEN} --out-dir "${SUR}"
    fi
    echo "=== $(date) | 2. VAE with the frozen surrogate decoder (${RUN}) ==="
    if [ -f "${RUN}/surrogateVAE_full_model.weights.h5" ]; then
        echo "    already trained: skipped"
    else
        python MODULES/comparison_surrogate_decoder/main_train_GCTN_surrogate.py --gpu "${GPU}" --data-dir "${DATA}" \
            --n-modes "${MODES}" --obs "${OBS}" --seed "${SEED}" --schedule plateau --epochs "${MAXEP}" \
            --surrogate-dir "${SUR}" --run-name "$(basename "${RUN}")"
    fi
    echo "=== $(date) | 3. post-processing and posterior quality (exact model) ==="
    python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}"
    python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${RUN}" --samples 500
fi
echo "=== $(date) | 4. comparison with the proposed model (${EIGEN}) ==="
if [ ! -f "${RUN}/Posterior_quality/ELBO_summary.json" ]; then
    echo "    the surrogate-decoder run is not complete yet (${RUN}): its last file, Posterior_quality/ELBO_summary.json,"
    echo "    is written by step 3, which runs on the CPU only (post-processing, then posterior quality)."
    echo "    Status files in the run folder: $(ls "${RUN}" 2>/dev/null | grep -E 'POSTPROCESSING|Problem_info|weights' | tr '\n' ' ')"
    echo "    Check the end of its log; the comparison runs automatically at the end of the full pipeline."
elif [ -f "${EIGEN}/Posterior_quality/ELBO_summary.json" ]; then
    python MODULES/comparison_surrogate_decoder/compare_eigen_vs_surrogate.py --eigen-run "${EIGEN}" --surrogate-run "${RUN}" \
        --out-dir "${OUT}/comparison${VAR}"
else
    echo "    the proposed run is not available yet. Train it with:"
    echo "    bash MODULES/model_comparison/run_convergence_gpu.sh --family gctn --obs ${OBS} --modes ${MODES} --gpu ${GPU}"
    echo "    and then: bash MODULES/comparison_surrogate_decoder/run_surrogate_comparison_gpu.sh --compare --targets ${TARGETS} --hidden \"${HIDDEN}\" --obs ${OBS} --modes ${MODES}"
fi
echo "=== $(date) | done ==="
