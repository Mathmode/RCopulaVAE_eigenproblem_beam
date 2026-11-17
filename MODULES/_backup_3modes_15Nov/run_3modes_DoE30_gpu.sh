#!/usr/bin/env bash
# R3.4 (data scarcity, fewer modes): GC-LogN trained with only the first 3 identified modes on the DoE30 data
# generation strategy, compared with the reference run (5 modes, GCLogN_13Nov-DoE30_7D_Gamma1.0_Conv6000).
# Same structures, damage scenarios, as-built masses, sensors (5), noise levels, likelihood calibration procedure,
# training script and settings; only the number of modes changes (the likelihood is calibrated for 3 modes).
#
# Stages (default all; finished steps are skipped, so the script can be relaunched):
#   data   data set Data/13Nov_FINAL_SlabS10_7D_DoE30_3modes (+ test dataset 'test_fine_uniform'); CPU only
#   train  training of the 3-mode model and full post-processing (benchmarks with SMC reference posteriors, violins)
#   ref    post-processing of the test dataset 'test_fine_uniform' for the 5-mode reference run (it was trained before
#          that split existed), so that both models are compared on the same 50-element test dataset
# Options: --gpu N (default 0), --stage data|train|ref|all
# Run from the project root with the environment active, e.g.:
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_3modes_DoE30_gpu.sh --gpu 0 --stage all > DoE30_3modes.log 2>&1 &
#   nohup bash MODULES/baseline_GaussCopula_GaussMarginals/run_3modes_DoE30_gpu.sh --stage ref > DoE30_ref_testA.log 2>&1 &
set -e
GPU=0
STAGE=all
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --stage) STAGE=$2; shift 2 ;;
        *) echo "unknown option '$1' (options: --gpu N --stage data|train|ref|all)"; exit 1 ;;
    esac
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi
stage() { [ "${STAGE}" = all ] || [ "${STAGE}" = "$1" ]; }

DATA5=Data/13Nov_FINAL_SlabS10_7D_DoE30
DATA=Data/13Nov_FINAL_SlabS10_7D_DoE30_3modes
VERSION=13Nov-DoE30-3modes_MolinerS10_kappa005_5sensors_3modes_mass5_localizedTestB
ROOT=Output/baseline_GaussCopula_GaussMarginals
REF=${ROOT}/GCLogN_13Nov-DoE30_7D_Gamma1.0_Conv6000
RUN=${ROOT}/GCLogN_13Nov-DoE30-3modes_7D_Gamma1.0_Conv6000

if stage data; then
    echo "=== $(date) | data set ${DATA} (3 modes) ==="
    if [ -f "${DATA}/alphas_test_fine.npy" ]; then echo "    already generated: skipped"; else
        if [ -d "${DATA}" ]; then echo "    ${DATA} exists but is incomplete: move or delete it first"; exit 1; fi
        python MODULES/review29oct_fixed/dataset_generation.py --data-dir "${DATA}" --n-modes 3 --version "${VERSION}"
    fi
    if [ -f "${DATA}/alphas_test_fine_uniform.npy" ]; then echo "    test_fine_uniform already added: skipped"; else
        python MODULES/review29oct_fixed/add_test_fine_uniform.py --data-dir "${DATA}"
    fi
    python - <<PY
import numpy as np
for s in ("train", "val", "test"):
    a = np.load("${DATA5}/alphas_" + s + ".npy"); b = np.load("${DATA}/alphas_" + s + ".npy")
    print(f"    {s}: same damage scenarios as the 5-mode data set: {a.shape == b.shape and np.array_equal(a, b)}")
PY
fi

if stage train; then
    [ -f "${DATA}/alphas_test_fine_uniform.npy" ] || { echo "run --stage data first"; exit 1; }
    echo "=== $(date) | GPU ${GPU} | 3-mode model ${RUN} ==="
    if [ -f "${RUN}/final_model_weights.weights.h5" ]; then echo "    already trained: skipped"; else
        python MODULES/baseline_GaussCopula_GaussMarginals/main_train_GCLogN.py --data-dir "${DATA}" --gpu "${GPU}" \
            --n-modes 3 --obs fv --seed 1234 --schedule plateau --epochs 6000 --run-name "$(pwd)/${RUN}"
    fi
    if [ -f "${RUN}/POSTPROCESSING_DONE.txt" ]; then echo "    post-processing done: skipped"; else
        python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}"
    fi
fi

if stage ref; then
    echo "=== $(date) | 5-mode reference ${REF}: test dataset test_fine_uniform ==="
    if [ -f "${REF}/POSTPROCESSING_test_fine_uniform_DONE.txt" ]; then echo "    already done: skipped"; else
        [ -f "${REF}/Test_predicted_props_test_fine_uniform.npy" ] || \
            python MODULES/review29oct_fixed/predict_split.py --run "${REF}" --split test_fine_uniform
        python MODULES/review29oct_fixed/postprocess_split.py --run-folder "${REF}" --split test_fine_uniform
    fi
fi
echo "=== $(date) | done (stage ${STAGE}) ==="
