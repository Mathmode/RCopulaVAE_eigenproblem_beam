#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/main_train_GCTN_surrogate.py

STEP 2 of the comparison with the Gaussian copula VAE of Ref. 41 (R1.4): trains the inverse model (encoder) with the
pre-trained surrogate decoder of step 1 attached in place of the eigenvalue solver, its weights frozen.

The training is that of the proposed model (review29oct_fixed/training_common.train_and_save): same data and splits, same
encoder, posterior family (Gaussian copula + truncated Gaussian marginals), calibrated likelihood, gamma = 1, warm-up,
optimizer and train-to-convergence rule (--schedule plateau). The only other change is unavoidable: the validation -ELBO
that selects the weights during training is computed with the surrogate, because a method that uses a surrogate decoder has
no eigenvalue solver available. Every evaluation afterwards (postprocessing.py, evaluate_posterior_quality.py) uses the
exact model, so both approaches are judged against the same exact posterior.

Files written in the run folder (in addition to those of train_and_save):
  surrogateVAE_full_model.weights.h5   encoder + frozen surrogate
  final_model_weights.weights.h5       the same encoder inside a GC-TN model (eigenvalue decoder unused), so that every
                                       tool that rebuilds a GC-TN run (load_trained_model) works unchanged
  Problem_info.npy / run_config.json   family GC-TN, decoder "surrogate", surrogate folder, parameters and training time

Run from the project root (after step 1):
    python MODULES/comparison_surrogate_decoder/main_train_GCTN_surrogate.py --gpu 0 --schedule plateau --epochs 6000 \
        --surrogate-dir Output/comparison_surrogate_decoder/surrogate_decoder_06Nov-FINAL5modes
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

from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.training_common import add_training_arguments, select_gpu

OUTPUT_DIR = os.path.join(C.PROJECT_ROOT, "Output", "comparison_surrogate_decoder")

parser = add_training_arguments(argparse.ArgumentParser(
    description="Step 2: Gaussian copula VAE with the pre-trained, frozen surrogate decoder (Ref. 41)."))
parser.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128], help="hidden layers of the encoder")
parser.add_argument("--surrogate-dir", type=str, required=True, help="folder written by train_surrogate_decoder.py")
args, _ = parser.parse_known_args()
select_gpu(args.gpu)   # before TensorFlow is imported


def build_from_info(cls, info):
    """Model of class cls rebuilt from Problem_info (as training_common.load_trained_model), called once to build it."""
    import numpy as np
    from MODULES.review29oct_fixed.preprocessing_and_eigen import load_baseline_matrices, load_problem_config
    pcfg = load_problem_config(info["data_dir"])
    _, K_basis, L_inv = load_baseline_matrices(info["data_dir"])
    n_modes, n_sens = int(pcfg["n_modes"]), len(pcfg["sensor_positions"])
    model = cls(input_dim=int(info["input_dim_enc"]), num_dofs=K_basis.shape[-1], n_modes=n_modes, Ke_matrices=K_basis,
                L_inv=L_inv, n_dims=int(info["n_dims"]), num_KS=1, num_samples=int(info["n_samples"]),
                gamma=float(info["gamma"]), mean_f=info["mean_f"], std_f=info["std_f"], lbound=float(info["lbound"]),
                likelihood=info["likelihood"], obs_types=info["obs_types"], sensor_nodes=pcfg.get("sensor_nodes"))
    unit = np.full((2, n_modes, n_sens), 1.0 / np.sqrt(n_sens), np.float32)
    _ = model((np.zeros((2, n_modes), np.float32), unit, unit, np.ones((2, int(info["n_dims"])), np.float32)))
    return model


def main():
    import numpy as np
    from MODULES.review29oct_fixed import training_common as TC
    from MODULES.baseline_GaussCopula_GaussMarginals.GCTN_model import GaussCopulaTNVAE
    from MODULES.comparison_surrogate_decoder.GCTN_surrogate_model import GaussCopulaTNSurrogateVAE
    from MODULES.comparison_surrogate_decoder.surrogate_decoder import SurrogateBridge, read_config

    sur_dir = args.surrogate_dir if os.path.isabs(args.surrogate_dir) else os.path.join(C.PROJECT_ROOT, args.surrogate_dir)
    sur_cfg = read_config(sur_dir)
    GaussCopulaTNSurrogateVAE.SURROGATE_DIR = sur_dir
    GaussCopulaTNSurrogateVAE.HIDDEN = tuple(args.hidden)
    GaussCopulaTNVAE.HIDDEN = tuple(args.hidden)

    def surrogate_validation_monitor(model, val_inputs, pred_keys, pcfg, data_dir, obs_types, gamma2, n_samples, seed=0):
        """The monitor of training_common.make_validation_monitor (same estimator and random numbers), with the surrogate
        as forward operator: a surrogate-based method has no eigenvalue solver to select its weights with."""
        import tensorflow as tf
        from MODULES.review29oct_fixed.posterior_families import family_from_preds
        from MODULES.review29oct_fixed.posterior_quality import mean_neg_elbo
        inputs = tuple(tf.convert_to_tensor(x) for x in val_inputs)
        obs = tuple(np.load(os.path.join(data_dir, f"{k}_val.npy")).astype(np.float64) for k in ("freqs", "vertmodes", "rotmodes"))
        bridge, lbound = SurrogateBridge(pcfg, sur_dir), float(pcfg["lbound"])

        def monitor():
            outs = [o.numpy() for o in model.Encoder_model(inputs)]
            return mean_neg_elbo(family_from_preds(dict(zip(pred_keys, outs)), lbound), obs, bridge, obs_types, gamma2,
                                 n_samples, seed)
        return monitor

    TC.make_validation_monitor = surrogate_validation_monitor
    extra = {"decoder": "surrogate NN (pre-trained, frozen; architecture of Ref. 41)", "surrogate_dir": sur_dir,
             "surrogate_parameters": sur_cfg.get("trainable_parameters"),
             "surrogate_training_time_sec": sur_cfg.get("training_time_sec"),
             "variational_family": "Gaussian copula + truncated-Gaussian marginals (as the proposed model)",
             "hidden": args.hidden,
             "weight_selection": "validation -ELBO computed with the surrogate (no eigenvalue solver in this approach)"}
    folder = TC.train_and_save(args, GaussCopulaTNSurrogateVAE, family="GC-TN", n_components=1, output_dir=OUTPUT_DIR,
                               run_prefix="GCTN-SurrogateDecoder", pred_keys=GaussCopulaTNVAE.PRED_KEYS, extra_info=extra)

    # --- weights: full model kept apart; the encoder inside a GC-TN model for the standard tools ---
    full = os.path.join(folder, "final_model_weights.weights.h5")
    full_sur = os.path.join(folder, "surrogateVAE_full_model.weights.h5")
    os.replace(full, full_sur)
    info = np.load(os.path.join(folder, "Problem_info.npy"), allow_pickle=True).item()
    sur_model = build_from_info(GaussCopulaTNSurrogateVAE, info)
    sur_model.load_weights(full_sur)
    plain = build_from_info(GaussCopulaTNVAE, info)
    plain.Encoder_model.set_weights(sur_model.Encoder_model.get_weights())
    plain.save_weights(full)

    # check: the GC-TN copy reproduces the saved encoder outputs of test set A
    from MODULES.review29oct_fixed.preprocessing_and_eigen import load_bridge_data
    data = load_bridge_data(info["data_dir"])
    test_inputs = data[8:12]
    saved = np.load(os.path.join(folder, "Test_predicted_props.npy"), allow_pickle=True).item()
    outs = plain.Encoder_model.predict(test_inputs, batch_size=256, verbose=0)
    err = max(float(np.max(np.abs(o - saved[k]))) for k, o in zip(GaussCopulaTNVAE.PRED_KEYS, outs))
    info.update({"weights_note": "final_model_weights.weights.h5 = encoder in a GC-TN model (decoder unused); "
                                 "surrogateVAE_full_model.weights.h5 = encoder + frozen surrogate",
                 "encoder_copy_max_abs_diff": err})
    np.save(os.path.join(folder, "Problem_info.npy"), info, allow_pickle=True)
    print(f"Encoder copied into a GC-TN model (max |difference| of the test outputs {err:.2e}). Run folder: {folder}")
    if err > 1e-4:
        raise RuntimeError("the GC-TN copy of the encoder does not reproduce the saved predictions")
    return folder


if __name__ == "__main__":
    main()
