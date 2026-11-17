#!/usr/bin/env bash
# Realistic design of experiments (13 Nov 2026, "DoE2", R1.6): healthy 20 %, one damaged component 45 %, two 20 %,
# three 5 %, uniform over the whole parameter space 10 % (config.py, CVAE_DOE=realistic). Structure, sensors, noise,
# modelling errors, prior, posterior model (GC-LogN) and training settings are those of the reference run
# GCLogN_06Nov-5modes_7D_Gamma1.0_Conv6000; only the damage scenarios of the data change.
#
# Stages, in order of priority (default: all of them, one after the other, on the same GPU):
#   data       data set Data/13Nov_FINAL_SlabS10_7D_DoE2 (training, validation, test sets A and B; skipped if it exists)
#   eigen      proposed model: GC-LogN with the eigenvalue solver (R1.3 Table R2, R1.6), post-processing, posterior quality
#   surrogate  Ref. 41: surrogate decoder + GC-LogN with the frozen surrogate, full training set (R1.4 Table R3, 14,000)
#   budget     500 training simulations: eigenvalue solver vs surrogate (R1.4 Table R3, 500)
# Every training is skipped if its weights already exist, so the script can be relaunched after an interruption.
#
# Options:
#   --gpu N        GPU index as in nvidia-smi (default 0)
#   --stage S      data | eigen | surrogate | budget | all (default all)
# Run from the project root (the folder that contains MODULES/ and Data/), with the environment active:
#   nohup bash MODULES/review29oct_fixed/run_DoE2_gpu.sh --gpu 0 > DoE2.log 2>&1 &
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
export CVAE_DOE=realistic
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi

DATA=Data/13Nov_FINAL_SlabS10_7D_DoE2
EIGEN=Output/baseline_GaussCopula_GaussMarginals/GCLogN_13Nov-DoE2_7D_Gamma1.0_Conv6000
S=Output/comparison_surrogate_decoder/DoE2
B=${S}/data_budget
COMMON=(--gpu "${GPU}" --n-modes 5 --obs fv --seed 1234 --schedule plateau --epochs 6000)
stage() { [ "${STAGE}" = all ] || [ "${STAGE}" = "$1" ]; }
post() {
    python MODULES/review29oct_fixed/postprocessing.py --run-folder "$1"
    python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "$1" --samples 500
}
need_data() {
    if [ ! -f "${DATA}/alphas_test_fine.npy" ]; then echo "data set ${DATA} not found: run --stage data first"; exit 1; fi
}
echo "=== $(date) | GPU ${GPU}: $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null) | stage ${STAGE} ==="

if stage data; then
    echo "=== $(date) | data set ${DATA} ==="
    if [ -f "${DATA}/alphas_test_fine.npy" ]; then echo "    already generated: skipped"; else
        python MODULES/review29oct_fixed/dataset_generation.py --data-dir "${DATA}"
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
