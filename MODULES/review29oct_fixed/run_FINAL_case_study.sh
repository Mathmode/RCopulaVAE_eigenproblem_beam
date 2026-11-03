#!/usr/bin/env bash
# FINAL case study (3 Nov 2026): girder line of a short-span composite bridge on elastomeric pads, 5 sensors,
# 5 % mass scatter, 2.5 % within-zone stiffness, 2.5 % / 5 % noise, no temperature; test set B = 50-element model with
# damage localized inside the zones. Steps: training of the proposed model on one GPU, post-processing of both test
# sets (paper metrics, reliability diagrams, copula correlations, benchmarks VAE vs reference), posterior quality,
# localized-damage scenarios, showcase with replicas, distribution-shift study (R1.5).
#
# Options (all optional):
#   --gpu N     GPU index as in nvidia-smi (default 2)
#   --ks K      Kumaraswamy components per marginal (default 3)
#   --modes M   identified modes, 1-5 (default 5); M < 5 uses the first M modes of the same observations
#               (data set ..._<M>modes, created once by derive_mode_subset.py)
#   --post      skip the training, only post-process the run
# Run from the project root (the folder that contains MODULES/ and Data/), with your environment active:
#     nohup bash MODULES/review29oct_fixed/run_FINAL_case_study.sh --gpu 2 --ks 3 --modes 5 > case_study_FINAL.log 2>&1 &
set -e
GPU=2
KS=3
MODES=5
MODE=all
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --ks) KS=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        --post) MODE=post; shift ;;
        *) echo "unknown option '$1' (options: --gpu N --ks K --modes M --post)"; exit 1 ;;
    esac
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# use the C++ runtime of the conda environment (avoids "CXXABI_1.3.15 not found" when matplotlib is imported)
if [ -n "${CONDA_PREFIX}" ]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi
DATA=Data/03Nov_FINAL_GirderBearings_7D_5sensors_5modes
TAG=03Nov-FINAL${MODES}modes          # the run name always states the number of modes
RUN=KSVAE_fixed_${TAG}_7D_${KS}KS_obs-fv_Gamma1.0_Ep1500
echo "=== $(date) | GPU ${GPU} | K = ${KS} Kumaraswamy components | ${MODES} modes | run ${RUN} ==="

if [ "${MODE}" != "post" ]; then
    echo "=== $(date) | training ==="
    python MODULES/review29oct_fixed/main_train_KSVAE.py --gpu "${GPU}" --epochs 1500 --data-dir "${DATA}" \
        --n-modes "${MODES}" --num-ks "${KS}" --run-name "${RUN}"
fi
echo "=== $(date) | post-processing (test sets A and B) ==="
python MODULES/review29oct_fixed/postprocessing.py --run-folder "${RUN}"
echo "=== $(date) | posterior quality (validation -ELBO, Pareto-k; also the entry of this K in the K study) ==="
python MODULES/model_comparison/evaluate_posterior_quality.py --run-folder "${RUN}" --samples 500
echo "=== $(date) | localized-damage scenarios ==="
python MODULES/review29oct_fixed/localized_damage_study.py --run-folder "${RUN}"
echo "=== $(date) | showcase scenarios with replicas ==="
python MODULES/review29oct_fixed/showcase_study.py --run-folder "${RUN}"
echo "=== $(date) | distribution shift (R1.5: six shifted test sets, exact posteriors of 40 cases each) ==="
python MODULES/review29oct_fixed/ood_study.py --run-folder "${RUN}"
echo "=== $(date) | done: Output/review29oct_fixed/${RUN} ==="
