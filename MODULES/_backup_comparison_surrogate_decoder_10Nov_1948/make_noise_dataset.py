#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/make_noise_dataset.py

R1.4, noise-precision study: the final data set of the revision with another identification noise on the frequencies
(and, optionally, on the mode shapes). Only config.FREQ_NOISE / config.MODE_NOISE change, in this process; everything else
is generated exactly as the final data set (same seeds: same damage states, as-built structures, localization of the test
damage and standard-normal noise draws, scaled by the new level), and the likelihood is calibrated again with the new
noise (dataset_generation.py reads the noise levels at call time; the reference posteriors read the calibrated likelihood
from problem_config.json).

Everything runs inside main(): the reference posteriors use a multiprocessing pool, whose workers import this file again
when processes are spawned (Windows, or Python >= 3.14 on Linux); they must not parse the command line.

    python MODULES/comparison_surrogate_decoder/make_noise_dataset.py --freq-noise 0.01 \
        --data-dir Data/06Nov_FINAL_SlabS10_7D_5sensors_5modes_fnoise1pct \
        --version 06Nov-Fnoise1pct_MolinerS10_kappa005_5sensors_5modes_mass5_localizedTestB
"""
import argparse
import os
import sys

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    from MODULES.review29oct_fixed import config as C

    parser = argparse.ArgumentParser(description="Final data set with another identification noise (noise-precision study).")
    parser.add_argument("--freq-noise", type=float, required=True,
                        help="relative std of the identified frequencies (0.025 in the final data set)")
    parser.add_argument("--mode-noise", type=float, default=None,
                        help="std of the mode-shape components relative to the peak (default: config, 0.05)")
    args, rest = parser.parse_known_args()
    if not any(a.startswith("--data-dir") for a in rest):
        sys.exit("--data-dir is required: the final data set must not be overwritten")
    C.FREQ_NOISE = args.freq_noise
    if args.mode_noise is not None:
        C.MODE_NOISE = args.mode_noise
    print(f"identification noise: frequencies {100 * C.FREQ_NOISE:.2f} %, mode shapes {100 * C.MODE_NOISE:.2f} % of the peak",
          flush=True)

    from MODULES.review29oct_fixed import dataset_generation as G   # after the noise levels are set

    sys.argv = [sys.argv[0]] + rest
    G.main()


if __name__ == "__main__":
    main()
