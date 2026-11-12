#!/usr/bin/env bash
# R1.4: GC-LogN VAE with a neural-network decoder trained JOINTLY with the encoder (one step, random initial weights, no
# pre-training, no damage labels), to compare with the proposed model GCLogN_06Nov-5modes_7D_Gamma1.0_Conv6000.
# Same data, encoder, likelihood, seed and convergence rule (plateau); cap of 10 000 epochs (--max-epochs to change it).
#   1. training                       -> Output/comparison_surrogate_decoder/VAE_NNdec
#   2. post-processing (test sets A and B, benchmarks P0-P10 with --single-corners, violins), as the proposed model
#   3. posterior quality (closeness to the exact posterior)
#   4. the six shifted test sets (ood_study.py, exact posterior of the first 5 cases of every set)
# Run from the project root, with the environment active:
#   nohup bash MODULES/comparison_surrogate_decoder/run_NNdec_gpu.sh --gpu 0 > NNdec.log 2>&1 &
set -e
GPU=0
MAXEP=10000
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --max-epochs) MAXEP=$2; shift 2 ;;
        *) echo "unknown option '$1' (options: --gpu N --max-epochs E)"; exit 1 ;;
    esac
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi
DATA=Data/06Nov_FINAL_SlabS10_7D_5sensors_5modes
RUN=Output/comparison_surrogate_decoder/VAE_NNdec

echo "=== $(date) | GPU ${GPU}: $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null) | ${RUN} ==="
echo "=== $(date) | 1. training (cap ${MAXEP} epochs) ==="
if [ -f "${RUN}/jointVAE_full_model.weights.h5" ]; then echo "    already trained: skipped"; else
    python MODULES/comparison_surrogate_decoder/main_train_joint_NN.py --gpu "${GPU}" --data-dir "${DATA}" --n-modes 5 \
        --obs fv --seed 1234 --schedule plateau --epochs "${MAXEP}" --run-name "$(pwd)/${RUN}"
fi
echo "=== $(date) | 2. post-processing ==="
python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}" --single-corners
echo "=== $(date) | 3. posterior quality ==="
python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${RUN}" --samples 500
echo "=== $(date) | 4. shifted test sets ==="
python MODULES/review29oct_fixed/ood_study.py --run-folder "${RUN}" --n-gt 5
echo "=== $(date) | done: ${RUN} ==="
