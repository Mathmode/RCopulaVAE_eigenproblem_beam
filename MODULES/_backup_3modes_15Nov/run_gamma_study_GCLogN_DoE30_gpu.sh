#!/usr/bin/env bash
# Sensitivity analysis to gamma (R3.2) for the reference model GC-LogN on the DoE30 data set
# (Data/13Nov_FINAL_SlabS10_7D_DoE30). Same script, data, likelihood, warm-up, seed and convergence rule as the
# reference run GCLogN_13Nov-DoE30_7D_Gamma1.0_Conv6000 (run_GCLogN_DoE30_gpu.sh --stage eigen); only gamma changes.
#   loss = tau * E_q[NLL] - gamma^2 * H[q]; gamma = 1 is the negative ELBO, gamma != 1 targets the tempered posterior.
# gamma = 1.0 is the existing reference run and is never retrained here.
#
# Steps per gamma (finished steps are skipped, so the script can be relaunched):
#   1. training   2. post-processing (test sets, SMC reference posteriors at the same gamma)
#   3. posterior quality (validation set: ELBO, calibration)
# Then, with --compare (once every gamma has finished): tables and figure across gamma, written to
#   Output/baseline_GaussCopula_GaussMarginals/gamma_study_DoE30
#
# Options:
#   --gpu N            GPU index as in nvidia-smi (default 0)
#   --gammas "G ..."   gamma values to train (default "0.5 0.75 1.25 1.5 2.0")
#   --compare          only the comparison across gamma (no training)
# Run from the project root (the folder that contains MODULES/ and Data/), with the environment active.
# One gamma per GPU, for example:
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_gamma_study_GCLogN_DoE30_gpu.sh --gpu 0 --gammas "0.5"  > gamma_DoE30_0p5.log  2>&1 &
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_gamma_study_GCLogN_DoE30_gpu.sh --gpu 1 --gammas "0.75" > gamma_DoE30_0p75.log 2>&1 &
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_gamma_study_GCLogN_DoE30_gpu.sh --compare > gamma_DoE30_compare.log 2>&1
set -e
GPU=0
GAMMAS="0.5 0.75 1.25 1.5 2.0"
MODE=train
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --gammas) GAMMAS=$2; shift 2 ;;
        --compare) MODE=compare; shift ;;
        *) echo "unknown option '$1' (options: --gpu N --gammas \"G ...\" --compare)"; exit 1 ;;
    esac
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi

DATA=Data/13Nov_FINAL_SlabS10_7D_DoE30
VERSION=13Nov-DoE30_MolinerS10_kappa005_5sensors_5modes_mass5_localizedTestB
ROOT=Output/baseline_GaussCopula_GaussMarginals
REF=${ROOT}/GCLogN_13Nov-DoE30_7D_Gamma1.0_Conv6000
OUT=${ROOT}/gamma_study_DoE30
COMMON=(--gpu "${GPU}" --n-modes 5 --obs fv --seed 1234 --schedule plateau --epochs 6000)
grep -q "\"${VERSION}\"" "${DATA}/dataset_version.json" 2>/dev/null || { echo "data set ${DATA} (version ${VERSION}) not found"; exit 1; }
[ -f "${REF}/final_model_weights.weights.h5" ] || { echo "reference run ${REF} (gamma = 1) not found"; exit 1; }

if [ "${MODE}" = train ]; then
    echo "=== $(date) | GPU ${GPU}: $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null) | gammas ${GAMMAS} ==="
    for G in ${GAMMAS}; do
        G_FMT=$(python -c "print(float('${G}'))")
        if [ "${G_FMT}" = "1.0" ]; then echo "    gamma = 1.0 is the reference run ${REF}: skipped"; continue; fi
        RUN=${ROOT}/GCLogN_13Nov-DoE30_7D_Gamma${G_FMT}_Conv6000
        echo "=== $(date) | gamma = ${G_FMT}: ${RUN} ==="
        if [ -f "${RUN}/final_model_weights.weights.h5" ]; then echo "    1. already trained: skipped"; else
            python MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCLogN.py --data-dir "${DATA}" "${COMMON[@]}" \
                --gamma "${G_FMT}" --run-name "$(pwd)/${RUN}"
        fi
        if [ -f "${RUN}/POSTPROCESSING_DONE.txt" ]; then echo "    2. post-processing done: skipped"; else
            python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}"
        fi
        if [ -f "${RUN}/Posterior_quality/ELBO_summary.json" ] && [ -f "${RUN}/Posterior_quality/Validation_set/Manuscript_Metrics.json" ]; then
            echo "    3. posterior quality done: skipped"; else
            python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${RUN}" --samples 500
        fi
    done
else
    for F in "${REF}/Posterior_quality/ELBO_summary.json" "${REF}/Posterior_quality/Validation_set/Manuscript_Metrics.json"; do
        [ -f "${F}" ] || python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${REF}" --samples 500
    done
    RUNS=()
    for D in ${ROOT}/GCLogN_13Nov-DoE30_7D_Gamma*_Conv6000; do
        if [ -f "${D}/Posterior_quality/ELBO_summary.json" ]; then RUNS+=("$(pwd)/${D}"); else echo "    not finished, left out: ${D}"; fi
    done
    echo "=== $(date) | comparison of ${#RUNS[@]} runs -> ${OUT} ==="
    python MODULES/baseline_GaussCopula_GaussMarginals/gamma_sensitivity_study.py --runs "${RUNS[@]}" --out-dir "${OUT}"
fi
echo "=== $(date) | done ==="
