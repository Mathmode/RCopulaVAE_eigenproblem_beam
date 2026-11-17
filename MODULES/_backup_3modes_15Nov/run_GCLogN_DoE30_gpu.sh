#!/usr/bin/env bash
# GC-LogN (Gaussian copula + truncated log-normal marginals, main_train_GCLogN.py of this folder) on the data generation
# strategy of 13 Nov 2026 ("DoE30", R1.6; review29oct_fixed/config.SCENARIO_FRACTIONS): healthy 30 %, one damaged
# component 30 %, two 15 %, three 10 %, four or more 5 %, uniform over the whole parameter space 10 %. Same training
# script, arguments and defaults as the reference run GCLogN_06Nov-5modes_7D_Gamma1.0_Conv6000 (trained with
# model_comparison/run_convergence_gpu.sh --family gclogn); structure, sensors, noise, modelling errors and prior are
# unchanged: only the damage scenarios of the data change. The data generator and the shared training loop live in
# review29oct_fixed (dataset_generation.py, training_common.py), as for every run of this folder.
#
# Stages, in order of priority (default: all of them, one after the other, on the same GPU):
#   data       data set Data/13Nov_FINAL_SlabS10_7D_DoE30 (training, validation, test sets A and B); CPU only
#   eigen      proposed model: GC-LogN with the eigenvalue solver, post-processing (all figures and metrics of test sets
#              A and B), posterior quality
#   surrogate  Ref. 41 (R1.4, Table R3): surrogate decoder + GC-LogN with the frozen surrogate, full training set
#   budget     Ref. 41 (R1.4, Table R3): eigenvalue solver vs surrogate with 500 training simulations
# Finished steps are skipped, so the script can be relaunched after an interruption.
#
# Options:
#   --gpu N        GPU index as in nvidia-smi (default 0)
#   --stage S      data | eigen | surrogate | budget | all (default all)
# Run from the project root (the folder that contains MODULES/ and Data/), with the environment active:
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_GCLogN_DoE30_gpu.sh --gpu 2 --stage data > DoE30_data.log 2>&1 &
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_GCLogN_DoE30_gpu.sh --gpu 2 --stage eigen > DoE30_eigen.log 2>&1 &
set -e
GPU=0
STAGE=all
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --stage) STAGE=$2; shift 2 ;;
        *) echo "unknown option '$1' (options: --gpu N --stage data|eigen|surrogate|budget|all)"; exit 1 ;;
    esac
done
case "${STAGE}" in data|eigen|surrogate|budget|all) ;; *) echo "--stage must be data, eigen, surrogate, budget or all"; exit 1 ;; esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi

DATA=Data/13Nov_FINAL_SlabS10_7D_DoE30
VERSION=13Nov-DoE30_MolinerS10_kappa005_5sensors_5modes_mass5_localizedTestB
EIGEN=Output/baseline_GaussCopula_GaussMarginals/GCLogN_13Nov-DoE30_7D_Gamma1.0_Conv6000
S=Output/comparison_surrogate_decoder/DoE30
B=${S}/data_budget
COMMON=(--gpu "${GPU}" --n-modes 5 --obs fv --seed 1234 --schedule plateau --epochs 6000)
stage() { [ "${STAGE}" = all ] || [ "${STAGE}" = "$1" ]; }
post() {
    python MODULES/review29oct_fixed/postprocessing.py --run-folder "$1"
    python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "$1" --samples 500
}
data_ok() { [ -f "${DATA}/alphas_test_fine.npy" ] && grep -q "\"${VERSION}\"" "${DATA}/dataset_version.json" 2>/dev/null; }
need_data() {
    if ! data_ok; then echo "data set ${DATA} (version ${VERSION}) not found or incomplete: run --stage data first"; exit 1; fi
}
echo "=== $(date) | GPU ${GPU}: $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null) | stage ${STAGE} ==="

if stage data; then
    echo "=== $(date) | data set ${DATA} ==="
    if data_ok; then echo "    already generated: skipped"; else
        if [ -d "${DATA}" ]; then
            echo "    ${DATA} exists but is incomplete or of another version: move or delete it first"; exit 1
        fi
        python MODULES/review29oct_fixed/dataset_generation.py --data-dir "${DATA}"
        data_ok || { echo "    data generation did not finish (no test set B or wrong version)"; exit 1; }
    fi
fi

if stage eigen; then
    need_data
    echo "=== $(date) | proposed model, GC-LogN + eigenvalue solver: ${EIGEN} ==="
    if [ -f "${EIGEN}/final_model_weights.weights.h5" ]; then echo "    already trained: skipped"; else
        python MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCLogN.py --data-dir "${DATA}" "${COMMON[@]}" \
            --run-name "$(pwd)/${EIGEN}"
    fi
    post "${EIGEN}"
fi

if stage surrogate; then
    need_data
    echo "=== $(date) | Ref. 41, full training set: surrogate decoder ${S}/surrogate ==="
    if [ -f "${S}/surrogate/surrogate_decoder.weights.h5" ]; then echo "    already trained: skipped"; else
        python MODULES/comparison_surrogate_decoder/train_surrogate_decoder.py --gpu "${GPU}" --data-dir "${DATA}" \
            --n-modes 5 --seed 1234 --targets nominal --hidden 10 30 50 70 80 --out-dir "${S}/surrogate"
    fi
    echo "=== $(date) | Ref. 41, full training set: GC-LogN with the frozen surrogate ${S}/LogN_surrogate ==="
    if [ -f "${S}/LogN_surrogate/surrogateVAE_full_model.weights.h5" ]; then echo "    already trained: skipped"; else
        python MODULES/comparison_surrogate_decoder/main_train_GCTN_surrogate.py --family gclogn --data-dir "${DATA}" \
            "${COMMON[@]}" --surrogate-dir "${S}/surrogate" --run-name "$(pwd)/${S}/LogN_surrogate"
    fi
    post "${S}/LogN_surrogate"
    if [ -f "${EIGEN}/Posterior_quality/ELBO_summary.json" ]; then
        python MODULES/comparison_surrogate_decoder/compare_eigen_vs_surrogate.py --eigen-run "${EIGEN}" \
            --surrogate-run "${S}/LogN_surrogate" --out-dir "${S}/LogN_comparison" --no-figure
    else
        echo "    proposed run not finished yet: run --stage eigen, then this stage again (trainings are skipped)"
    fi
fi

if stage budget; then
    need_data
    D=${B}/data_ntrain500
    echo "=== $(date) | 500 training simulations: reduced data set ${D} ==="
    if [ -f "${D}/alphas_train.npy" ]; then echo "    already generated: skipped"; else
        python MODULES/comparison_surrogate_decoder/surrogate_data_budget.py --data-dir "${DATA}" --sizes 500 \
            --out-dir "${B}" --make-data-only
    fi
    echo "=== $(date) | 500: GC-LogN + eigenvalue solver ${B}/LogN_eigen_ntrain500 ==="
    if [ -f "${B}/LogN_eigen_ntrain500/final_model_weights.weights.h5" ]; then echo "    already trained: skipped"; else
        python MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCLogN.py --data-dir "${D}" "${COMMON[@]}" \
            --run-name "$(pwd)/${B}/LogN_eigen_ntrain500"
    fi
    echo "=== $(date) | 500: surrogate decoder ${B}/surrogate_ntrain500 ==="
    if [ -f "${B}/surrogate_ntrain500/surrogate_decoder.weights.h5" ]; then echo "    already trained: skipped"; else
        python MODULES/comparison_surrogate_decoder/train_surrogate_decoder.py --gpu "${GPU}" --data-dir "${D}" \
            --n-modes 5 --seed 1234 --targets nominal --hidden 10 30 50 70 80 --out-dir "${B}/surrogate_ntrain500"
    fi
    echo "=== $(date) | 500: GC-LogN with the frozen surrogate ${B}/LogN_surrogate_ntrain500 ==="
    if [ -f "${B}/LogN_surrogate_ntrain500/surrogateVAE_full_model.weights.h5" ]; then echo "    already trained: skipped"; else
        python MODULES/comparison_surrogate_decoder/main_train_GCTN_surrogate.py --family gclogn --data-dir "${D}" \
            "${COMMON[@]}" --surrogate-dir "${B}/surrogate_ntrain500" --run-name "$(pwd)/${B}/LogN_surrogate_ntrain500"
    fi
    for RUN in "${B}/LogN_eigen_ntrain500" "${B}/LogN_surrogate_ntrain500"; do post "${RUN}"; done
    python MODULES/comparison_surrogate_decoder/compare_eigen_vs_surrogate.py --eigen-run "${B}/LogN_eigen_ntrain500" \
        --surrogate-run "${B}/LogN_surrogate_ntrain500" --out-dir "${B}/LogN_comparison_ntrain500" --no-figure
fi
echo "=== $(date) | done (stage ${STAGE}) ==="
