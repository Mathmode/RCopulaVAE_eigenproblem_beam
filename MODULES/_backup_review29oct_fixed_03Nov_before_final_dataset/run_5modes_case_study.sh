#!/usr/bin/env bash
# Definitive case study: 5 sensors, 5 modes, v6 settings (5 % mass scatter, 2.5 % within-zone stiffness, 2.5 % / 5 %
# noise, no temperature). Steps: training on one GPU, post-processing of both test sets (paper metrics, reliability
# diagrams, copula correlations, benchmarks VAE vs reference), localized-damage scenarios, showcase with replicas.
#
# Run from the project root (the folder that contains MODULES/ and Data/), with your environment active:
#     nohup bash MODULES/review29oct_fixed/run_5modes_case_study.sh 2 > case_study_5modes.log 2>&1 &
#     nohup bash MODULES/review29oct_fixed/run_5modes_case_study.sh 2 post > case_study_5modes_post.log 2>&1 &
# First argument: GPU number (default 2). Second argument "post": skip the training, only post-process the run.
set -e
GPU=${1:-2}
MODE=${2:-all}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# use the C++ runtime of the conda environment (avoids "CXXABI_1.3.15 not found" when matplotlib is imported)
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi
DATA=Data/01Nov_BridgeBearings_7D_v6_5sensors_5modes
RUN=KSVAE_fixed_01Nov-5modes_7D_3KS_obs-fv_Gamma1.0_Ep1500

if [ "${MODE}" != "post" ]; then
    echo "=== $(date) | training on GPU ${GPU} ==="
    python MODULES/review29oct_fixed/main_train_KSVAE.py --gpu "${GPU}" --epochs 1500 --data-dir "${DATA}"
fi
echo "=== $(date) | post-processing (test sets A and B) ==="
python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}"
echo "=== $(date) | localized-damage scenarios ==="
python MODULES/review29oct_fixed/localized_damage_study.py --run-folder "${RUN}"
echo "=== $(date) | showcase scenarios with replicas ==="
python MODULES/review29oct_fixed/showcase_study.py --run-folder "${RUN}"
echo "=== $(date) | done: Output/review29oct_fixed/${RUN} ==="
