#!/usr/bin/env bash
# Number-of-modes study (01Nov): 5 sensors with 5 modes and with 3 modes, definitive v6 settings
# (5 % mass scatter, 2.5 % within-zone stiffness, 2.5 % / 5 % noise, no temperature).
# For each data set: training on one GPU, full post-processing (metrics of the paper, reliability diagrams,
# corner plots for both test sets) and the localized-damage scenarios. Jobs run one after the other.
#
# Run from the project root (the folder that contains MODULES/ and Data/):
#     nohup bash MODULES/review29oct_fixed/run_01Nov_modes_study.sh 2 > 01Nov_modes_study.log 2>&1 &
# The argument is the GPU number (default 2). Results: Output/review29oct_fixed/KSVAE_fixed_01Nov-<n>modes_*/
set -e
GPU=${1:-2}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for MODES in 5 3; do
    DATA=Data/01Nov_BridgeBearings_7D_v6_5sensors_${MODES}modes
    RUN=KSVAE_fixed_01Nov-${MODES}modes_7D_3KS_obs-fv_Gamma1.0_Ep1500
    echo "=== $(date) | ${MODES} modes: training on GPU ${GPU} ==="
    python MODULES/review29oct_fixed/main_train_KSVAE.py --gpu "${GPU}" --epochs 1500 --data-dir "${DATA}"
    echo "=== $(date) | ${MODES} modes: post-processing ==="
    python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}"
    echo "=== $(date) | ${MODES} modes: localized-damage scenarios ==="
    python MODULES/review29oct_fixed/localized_damage_study.py --run-folder "${RUN}"
done
echo "=== $(date) | done ==="
