#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/variant_GaussCopula_GaussMixMarginals/main_train_GCGMix.py

Trains the VARIANT with truncated-Gaussian mixture marginals (Gaussian copula, K components per marginal) with the training
procedure of the proposed method (review29oct_fixed/training_common.py: same data, likelihood, gamma = 1, warm-up, optimizer
and samples of q; --schedule cosine or plateau).

Run from the project root, e.g. on GPU 2 of the server:
    python MODULES/variant_GaussCopula_GaussMixMarginals/main_train_GCGMix.py --gpu 2 --gmix 3 \
        --data-dir Data/03Nov_FINAL_GirderBearings_7D_5sensors_5modes --postprocess
Outputs: Output/variant_GaussCopula_GaussMixMarginals/GCGMix3_<data tag>_7D_obs-fv_Gamma1.0_Ep1500/
"""

import os
import sys
import argparse

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    THIS_DIR = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.training_common import add_training_arguments, select_gpu

OUTPUT_DIR = os.path.join(C.PROJECT_ROOT, "Output", "variant_GaussCopula_GaussMixMarginals")

parser = add_training_arguments(argparse.ArgumentParser(
    description="Variant: Gaussian copula + truncated-Gaussian mixture marginals (same ELBO and training as the KS model)."))
parser.add_argument("--gmix", type=int, default=3, help="truncated-Gaussian components per marginal")
args, _ = parser.parse_known_args()
select_gpu(args.gpu)   # before TensorFlow is imported


def main():
    from MODULES.review29oct_fixed.training_common import train_and_save
    from MODULES.variant_GaussCopula_GaussMixMarginals.GCGMix_model import GaussCopulaGMixVAE
    return train_and_save(args, GaussCopulaGMixVAE, family="GC-GMix", n_components=args.gmix, output_dir=OUTPUT_DIR,
                          run_prefix=f"GCGMix{args.gmix}", pred_keys=GaussCopulaGMixVAE.PRED_KEYS,
                          extra_info={"variational_family": f"Gaussian copula + mixture of {args.gmix} truncated "
                                                            f"Gaussians per marginal"})


if __name__ == "__main__":
    main()
