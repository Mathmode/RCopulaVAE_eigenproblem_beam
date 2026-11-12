#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/comparison_surrogate_decoder/main_train_joint_NN.py

R1.4: trains the GC-LogN VAE with a neural-network decoder trained jointly with the encoder (joint_NN_model.py): one
training step, random initial weights, no pre-training, no damage labels. Same data, encoder, likelihood, seed and
convergence rule as the proposed model (GCLogN_06Nov-5modes_7D_Gamma1.0_Conv6000); the validation monitor of the rule
uses the learned decoder (this approach has no eigenvalue solver).

Outputs (default Output/comparison_surrogate_decoder/VAE_NNdec):
  final_model_weights.weights.h5      the encoder inside a GC-LogN model, so post-processing works unchanged
  jointVAE_full_model.weights.h5      encoder + learned decoder
  nn_decoder.weights.h5, nn_decoder.json   the learned decoder alone, with its output scaling (for later checks)
  + every file of a standard run (predictions, history, Problem_info, run_config)

    python MODULES/comparison_surrogate_decoder/main_train_joint_NN.py --gpu 0 --schedule plateau --epochs 10000
"""

import argparse
import json
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
    parser = add_training_arguments(argparse.ArgumentParser(
        description="GC-LogN VAE with a NN decoder trained jointly with the encoder (no pre-training, no labels)."))
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 256, 128], help="hidden layers of the encoder")
    parser.add_argument("--dec-hidden", type=int, nargs="+", default=[10, 30, 50, 70, 80],
                        help="hidden layers of the decoder (default: architecture of Ref. 41)")
    args, _ = parser.parse_known_args()
    if args.run_name is None:
        args.run_name = "VAE_NNdec"
    select_gpu(args.gpu)                                             # before TensorFlow is imported

    import numpy as np
    from MODULES.review29oct_fixed import training_common as TC
    from MODULES.review29oct_fixed.fe_model import frequency_precision
    from MODULES.review29oct_fixed.preprocessing_and_eigen import load_problem_config, load_bridge_data
    from MODULES.baseline_GaussCopula_GaussMarginals.GCLogN_model import GaussCopulaLogNVAE
    from MODULES.comparison_surrogate_decoder.joint_NN_model import GaussCopulaLogNJointNN, LearnedDecoderBridge
    from MODULES.comparison_surrogate_decoder.surrogate_decoder import make_targets, MinMaxScaler

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(C.PROJECT_ROOT, args.data_dir)
    pcfg = load_problem_config(data_dir)
    # output scaling of the decoder from the MEASURED training data only (no simulation at known damage states)
    bias, _ = frequency_precision(pcfg["likelihood"])
    f_tr = np.load(os.path.join(data_dir, "freqs_train.npy"))
    v_tr = np.load(os.path.join(data_dir, "vertmodes_train.npy"))
    scaler = MinMaxScaler().fit(make_targets(f_tr, v_tr, bias))
    GaussCopulaLogNJointNN.SCALER_LO, GaussCopulaLogNJointNN.SCALER_HI = scaler.lo.tolist(), scaler.hi.tolist()
    GaussCopulaLogNJointNN.N_SENS = int(v_tr.shape[-1])
    GaussCopulaLogNJointNN.DEC_HIDDEN = tuple(args.dec_hidden)
    GaussCopulaLogNJointNN.HIDDEN = tuple(args.hidden)
    GaussCopulaLogNVAE.HIDDEN = tuple(args.hidden)

    def learned_decoder_monitor(model, val_inputs, pred_keys, pcfg_, data_dir_, obs_types, gamma2, n_samples, seed=0):
        """The monitor of training_common.make_validation_monitor (same estimator and random numbers), with the learned
        decoder as forward operator."""
        import tensorflow as tf
        from MODULES.review29oct_fixed.posterior_families import family_from_preds
        from MODULES.review29oct_fixed.posterior_quality import mean_neg_elbo
        inputs = tuple(tf.convert_to_tensor(x) for x in val_inputs)
        obs = tuple(np.load(os.path.join(data_dir_, f"{k}_val.npy")).astype(np.float64) for k in ("freqs", "vertmodes", "rotmodes"))
        bridge, lbound = LearnedDecoderBridge(model, pcfg_), float(pcfg_["lbound"])

        def monitor():
            outs = [o.numpy() for o in model.Encoder_model(inputs)]
            return mean_neg_elbo(family_from_preds(dict(zip(pred_keys, outs)), lbound), obs, bridge, obs_types, gamma2,
                                 n_samples, seed)
        return monitor

    TC.make_validation_monitor = learned_decoder_monitor
    extra = {"decoder": "NN trained jointly with the encoder (one step, random initial weights, no labels)",
             "decoder_hidden": list(args.dec_hidden), "hidden": args.hidden,
             "decoder_output_scaling": "range of the measured training data (log f - b, vertical shapes)",
             "variational_family": "Gaussian copula + truncated log-normal marginals (as the proposed model)",
             "weight_selection": "validation -ELBO computed with the learned decoder (no eigenvalue solver)"}
    folder = TC.train_and_save(args, GaussCopulaLogNJointNN, family="GC-LogN", n_components=1, output_dir=OUTPUT_DIR,
                               run_prefix="VAE_NNdec", pred_keys=GaussCopulaLogNVAE.PRED_KEYS, extra_info=extra)

    # --- weights: full model and decoder kept apart; the encoder inside a plain GC-LogN model for the standard tools ---
    full = os.path.join(folder, "final_model_weights.weights.h5")
    full_joint = os.path.join(folder, "jointVAE_full_model.weights.h5")
    os.replace(full, full_joint)
    info = np.load(os.path.join(folder, "Problem_info.npy"), allow_pickle=True).item()
    joint = build_from_info(GaussCopulaLogNJointNN, info)
    joint.load_weights(full_joint)
    joint.Decoder.save_weights(os.path.join(folder, "nn_decoder.weights.h5"))
    with open(os.path.join(folder, "nn_decoder.json"), "w", encoding="utf-8") as fh:
        json.dump({"hidden": list(args.dec_hidden), "n_in": int(info["n_dims"]), "n_modes": int(pcfg["n_modes"]),
                   "n_sensors": GaussCopulaLogNJointNN.N_SENS, "scaler_lo": scaler.lo.tolist(),
                   "scaler_hi": scaler.hi.tolist(), "lbound": float(info["lbound"]),
                   "outputs": "[log f, vertical shapes]; input u = (z - lbound) / (1 - lbound)"}, fh, indent=1)
    plain = build_from_info(GaussCopulaLogNVAE, info)
    plain.Encoder_model.set_weights(joint.Encoder_model.get_weights())
    plain.save_weights(full)

    data = load_bridge_data(info["data_dir"])
    saved = np.load(os.path.join(folder, "Test_predicted_props.npy"), allow_pickle=True).item()
    outs = plain.Encoder_model.predict(data[8:12], batch_size=256, verbose=0)
    err = max(float(np.max(np.abs(o - saved[k]))) for k, o in zip(GaussCopulaLogNVAE.PRED_KEYS, outs))
    info.update({"weights_note": "final_model_weights.weights.h5 = encoder in a GC-LogN model (decoder unused); "
                                 "jointVAE_full_model.weights.h5 = encoder + learned decoder",
                 "encoder_copy_max_abs_diff": err})
    np.save(os.path.join(folder, "Problem_info.npy"), info, allow_pickle=True)
    print(f"Encoder copied into a GC-LogN model (max |difference| of the test outputs {err:.2e}). Run folder: {folder}")
    if err > 1e-4:
        raise RuntimeError("the copy of the encoder does not reproduce the saved predictions")
    return folder


if __name__ == "__main__":
    main()
