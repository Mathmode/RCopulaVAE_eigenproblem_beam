#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct_fixed/training_common.py

Training procedure shared by the baselines of the comparison requested in R1.4 (MODULES/baseline_*): the same data,
prior, eigen-decoder, calibrated likelihood, gamma, likelihood warm-up, optimizer, learning-rate schedule, early
stopping on the validation -ELBO and outputs as main_train_KSVAE.py (step by step), so that every model differs from
the proposed one only in the family of q(z | m).

Usage from a baseline entry script (arguments are parsed BEFORE TensorFlow is imported, for the GPU selection):
    parser = add_training_arguments(argparse.ArgumentParser(...))
    args, _ = parser.parse_known_args()
    select_gpu(args.gpu)
    from MODULES.review29oct_fixed.training_common import train_and_save
    train_and_save(args, ModelClass, family="GC-TN", n_components=1, output_dir=..., run_prefix="GCTN",
                   pred_keys=[...])
"""

import os
import json
import re
import time

from MODULES.review29oct_fixed import config as C


def add_protocol_arguments(parser):
    """Options of the learning-rate protocol. Default: the fixed cosine schedule with early stopping on the validation
    -ELBO. --schedule plateau: train to convergence (GC_KSmix_models.ConvergenceController, the same rule for every
    family); --epochs is then only a cap and --patience is not used."""
    parser.add_argument("--schedule", choices=["cosine", "plateau"], default="cosine",
                        help="cosine: fixed budget of --epochs; plateau: train to convergence")
    parser.add_argument("--lr-patience", type=int, default=200,
                        help="plateau: epochs without improvement before halving the learning rate (or, at --lr-min, stopping)")
    parser.add_argument("--lr-min", type=float, default=1e-5, help="plateau: lowest learning rate (end of the cosine schedule)")
    parser.add_argument("--min-delta", type=float, default=0.002, help="plateau: smallest improvement that counts [nats]")
    parser.add_argument("--monitor-every", type=int, default=20, help="plateau: epochs between evaluations of the monitor")
    parser.add_argument("--monitor-samples", type=int, default=20,
                        help="plateau: samples of q per validation observation in the monitor")
    return parser


def schedule_tag(args):
    """Last token of the default run names: Ep<epochs> (fixed schedule) or Conv<cap> (train to convergence)."""
    return f"{'Conv' if args.schedule == 'plateau' else 'Ep'}{args.epochs}"


def make_optimizer(args, K, steps_per_epoch):
    """Adam with the cosine schedule of --epochs, or with a learning rate that ConvergenceController can change."""
    lr = args.lr
    if args.schedule == "cosine":
        lr = K.optimizers.schedules.CosineDecay(args.lr, decay_steps=args.epochs * steps_per_epoch, alpha=0.02)
    return K.optimizers.Adam(learning_rate=lr, global_clipnorm=10.0)


def make_validation_monitor(model, val_inputs, pred_keys, pcfg, data_dir, obs_types, gamma2, n_samples, seed=0):
    """
    Returns f() -> mean validation -ELBO of the current weights (unit space, as evaluate_posterior_quality.py), for any
    family: the numpy estimator of posterior_quality.py on the encoder outputs, with the same random numbers at every
    evaluation. The -ELBO logged by model.fit is too noisy to select weights: with an analytic entropy (GC-TN) its Monte
    Carlo error at fixed weights is about 0.018 nats, and a lucky dip can stop the training early.
    """
    import numpy as np
    import tensorflow as tf
    from MODULES.review29oct_fixed.fe_model import BridgeModel
    from MODULES.review29oct_fixed.posterior_families import family_from_preds
    from MODULES.review29oct_fixed.posterior_quality import mean_neg_elbo

    inputs = tuple(tf.convert_to_tensor(x) for x in val_inputs)
    obs = tuple(np.load(os.path.join(data_dir, f"{k}_val.npy")).astype(np.float64)
                for k in ("freqs", "vertmodes", "rotmodes"))
    bridge, lbound = BridgeModel(pcfg), float(pcfg["lbound"])

    def monitor():
        outs = [o.numpy() for o in model.Encoder_model(inputs)]
        return mean_neg_elbo(family_from_preds(dict(zip(pred_keys, outs)), lbound), obs, bridge, obs_types, gamma2,
                             n_samples, seed)
    return monitor


def make_training_callbacks(args, K, folder_path, monitor=None):
    """Returns the callback that keeps the best weights and the callback list of model.fit."""
    from MODULES.review29oct_fixed.GC_KSmix_models import (
        LikelihoodTempering, LearningRateLogger, FastInMemoryBestWeights, EpochTimer, ConvergenceController,
    )
    if args.schedule == "plateau":
        best_cb = ConvergenceController(monitor, args.warmup, args.monitor_every, args.lr_patience, args.lr_min,
                                        args.min_delta)
    else:
        best_cb = FastInMemoryBestWeights(monitor="val_neg_elbo", patience=args.patience, start_epoch=args.warmup)
    csv_cb = K.callbacks.CSVLogger(os.path.join(folder_path, "model_history.csv"))
    return best_cb, [LikelihoodTempering(args.tau0, args.warmup), LearningRateLogger(), EpochTimer(), best_cb, csv_cb]


def add_training_arguments(parser, default_samples=6):
    """The arguments (and defaults) of main_train_KSVAE.py, except --num-ks."""
    parser.add_argument("--gpu", type=str, default=None, help="GPU index as in nvidia-smi (default: leave unchanged)")
    parser.add_argument("--data-dir", type=str, default=C.DATA_DIR)
    parser.add_argument("--n-modes", type=int, default=None,
                        help="use the first M modes of the data set (subset folder created once; default: all)")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--warmup", type=int, default=150, help="epochs of likelihood tempering (tau: tau0 -> 1)")
    parser.add_argument("--tau0", type=float, default=0.01)
    parser.add_argument("--samples", type=int, default=default_samples, help="Monte-Carlo samples of q")
    parser.add_argument("--gamma", type=float, default=1.0, help="entropy weight gamma^2 (1 = exact Bayesian posterior)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--obs", type=str, default=None, help="observed quantities, e.g. fv (default: data set)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--postprocess", action="store_true", help="run postprocessing.py at the end")
    return add_protocol_arguments(parser)


def select_gpu(gpu):
    """Must run before TensorFlow is imported. GPUs are numbered as nvidia-smi does."""
    if gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def train_and_save(args, model_cls, family, n_components, output_dir, run_prefix, pred_keys, extra_info=None):
    """Trains model_cls (a subclass of My_CopulaKSVAE_withEigen) as main_train_KSVAE.py does and saves the same files
    (+ Val_predicted_props.npy). pred_keys: names of the encoder outputs in the prediction files. Returns the run
    folder."""
    import numpy as np
    import tensorflow as tf
    from MODULES.review29oct_fixed import tf_keras

    K = tf_keras()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)   # take GPU memory as needed (shared servers)
        except RuntimeError as e:
            print(f"GPU initialisation warning: {e}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set: all GPUs)')}")
    print(f"Devices: {[tf.config.experimental.get_device_details(g).get('device_name', g.name) for g in gpus] or 'CPU only'}")

    from MODULES.review29oct_fixed.preprocessing_and_eigen import (
        load_bridge_data, load_baseline_matrices, load_problem_config, load_test_split,
    )

    K.utils.set_random_seed(args.seed)
    K.backend.set_floatx("float32")

    # --- 1. DATA (same splits and log-frequency scaling as main_train_KSVAE.py) ---
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(C.PROJECT_ROOT, args.data_dir)
    if not os.path.exists(os.path.join(data_dir, "problem_config.json")):
        raise FileNotFoundError(f"no data set in {data_dir} (generate it with dataset_generation.py first)")
    if getattr(args, "n_modes", None) is not None:   # first M modes of the same observations
        from MODULES.review29oct_fixed.derive_mode_subset import derive_mode_subset
        data_dir = derive_mode_subset(data_dir, args.n_modes)
    pcfg = load_problem_config(data_dir)
    obs_types = args.obs or pcfg["obs_types"]
    lik = pcfg["likelihood"]
    (f_train, r_train, v_train, a_train,
     f_val, r_val, v_val, a_val,
     f_test, r_test, v_test, a_test,
     mean_f, std_f) = load_bridge_data(data_dir)
    _, K_basis, L_inv = load_baseline_matrices(data_dir)

    train_ds = tf.data.Dataset.from_tensor_slices(((f_train, r_train, v_train, a_train), a_train))
    train_ds = train_ds.cache().shuffle(len(f_train), seed=args.seed).batch(args.batch).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(((f_val, r_val, v_val, a_val), a_val))
    val_ds = val_ds.cache().batch(args.batch).prefetch(tf.data.AUTOTUNE)

    n_dims, n_modes, lbound = int(pcfg["n_dims"]), f_train.shape[1], float(pcfg["lbound"])
    n_dofs = K_basis.shape[-1]
    sizes = {"f": f_train.shape[1], "v": v_train.shape[1] * v_train.shape[2], "r": r_train.shape[1] * r_train.shape[2]}
    input_dim = sum(sizes[k] for k in obs_types)

    data_tag = str(pcfg.get("dataset_version", "")).split("_")[0]
    if not re.search(r"\d+modes$", data_tag):   # the run name always states the number of modes
        data_tag = f"{data_tag}{f_train.shape[1]}modes"
    run_name = args.run_name or (f"{run_prefix}_{data_tag + '_' if data_tag else ''}{n_dims}D_"
                                 f"obs-{obs_types}_Gamma{args.gamma}_{schedule_tag(args)}")
    folder_path = os.path.join(output_dir, run_name)
    os.makedirs(folder_path, exist_ok=True)

    # --- 2. MODEL ---
    model = model_cls(
        input_dim=input_dim, num_dofs=n_dofs, n_modes=n_modes, Ke_matrices=K_basis, L_inv=L_inv,
        n_dims=n_dims, num_KS=n_components, num_samples=args.samples, gamma=args.gamma,
        mean_f=mean_f, std_f=std_f, lbound=lbound,
        likelihood=lik, obs_types=obs_types, sensor_nodes=pcfg.get("sensor_nodes"),
    )
    steps_per_epoch = int(np.ceil(len(f_train) / args.batch))
    model.compile(optimizer=make_optimizer(args, K, steps_per_epoch))
    _ = model((f_train[:2], r_train[:2], v_train[:2], a_train[:2]))
    n_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))

    run_cfg = {k: v for k, v in vars(args).items()}
    run_cfg.update({"data_dir": data_dir, "obs_types": obs_types, "input_dim": input_dim, "run_name": run_name,
                    "family": family, "n_components": n_components, "trainable_parameters": n_params,
                    "likelihood": lik, "objective": "tau * E_q[NLL] - gamma^2 * H[q]  (uniform prior)",
                    **(extra_info or {})})
    with open(os.path.join(folder_path, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_cfg, fh, indent=2)

    # --- 3. TRAINING ---
    print(f"Training {family}: {n_dims}D, components={n_components}, S={args.samples}, gamma={args.gamma}, "
          f"obs={obs_types}, {n_params} trainable parameters, {args.schedule} schedule, {args.epochs} epochs "
          f"(warm-up {args.warmup}) -> {folder_path}")
    monitor = None
    if args.schedule == "plateau":
        monitor = make_validation_monitor(model, (f_val, r_val, v_val, a_val), pred_keys, pcfg, data_dir, obs_types,
                                          args.gamma ** 2, args.monitor_samples)
    best_cb, callbacks = make_training_callbacks(args, K, folder_path, monitor)
    start_time = time.time()
    history = model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, verbose=2, callbacks=callbacks)
    monitor_time = getattr(best_cb, "monitor_time", 0.0)
    training_time = time.time() - start_time - monitor_time
    n_run = len(history.history["loss"])
    print(f"Total training time: {training_time:.1f} s ({training_time / n_run:.2f} s/epoch)")

    model.save_weights(os.path.join(folder_path, "final_model_weights.weights.h5"))
    np.save(os.path.join(folder_path, "model_history.npy"), history.history, allow_pickle=True)

    # --- 4. POSTERIOR PARAMETERS OF THE TEST SETS AND OF THE VALIDATION SET ---
    splits = {"test": (f_test, r_test, v_test, a_test)}
    for split in pcfg.get("test_sets", {}):
        if split != "test":
            splits[split] = load_test_split(data_dir, split, mean_f, std_f)
    splits["val"] = (f_val, r_val, v_val, a_val)
    for split, inputs in splits.items():
        outs = model.Encoder_model.predict(inputs, batch_size=args.batch, verbose=0)
        name = {"test": "Test_predicted_props.npy", "val": "Val_predicted_props.npy"}.get(
            split, f"Test_predicted_props_{split}.npy")
        np.save(os.path.join(folder_path, name), dict(zip(pred_keys, outs)), allow_pickle=True)

    Problem_info = {
        "input_dim_enc": input_dim, "n_dims": n_dims, "n_zones": int(pcfg["n_zones"]), "family": family,
        "n_components": n_components, "num_KS": n_components, "trainable_parameters": n_params,
        "n_samples": args.samples, "epochs": args.epochs, "epochs_run": n_run, "warmup": args.warmup,
        "best_epoch": best_cb.best_epoch, "gamma": args.gamma, "std_f": std_f, "mean_f": mean_f, "lbound": lbound,
        "training_time_sec": training_time, "data_dir": data_dir, "obs_types": obs_types, "likelihood": lik,
        "test_sets": [s for s in splits if s != "val"], "seed": args.seed, "schedule": args.schedule,
        "monitor_time_sec": monitor_time, "converged": getattr(best_cb, "converged", None), **(extra_info or {}),
    }
    np.save(os.path.join(folder_path, "Problem_info.npy"), Problem_info, allow_pickle=True)
    print(f"Saved weights, history, posterior parameters (test sets + validation) and problem info to {folder_path}")

    if args.postprocess:
        from MODULES.review29oct_fixed.postprocessing import run_postprocessing
        run_postprocessing(folder_path)
    return folder_path


KS_PRED_KEYS = ["test_a", "test_b", "test_weight_vals", "test_offdiag_elems", "test_diag_elems"]


def load_trained_model(run_folder, info):
    """Rebuilds the model of a trained run of any family (proposed KS model or a baseline of MODULES/baseline_*) from
    its Problem_info and loads the weights (also files written by another Keras version). Returns the model and the
    names of its encoder outputs in the prediction files."""
    import numpy as np
    from MODULES.review29oct_fixed import tf_keras
    K = tf_keras()
    K.backend.set_floatx("float32")
    from MODULES.review29oct_fixed.preprocessing_and_eigen import load_baseline_matrices, load_problem_config
    from MODULES.review29oct_fixed.GC_KSmix_models import My_CopulaKSVAE_withEigen, load_vae_weights

    family = info.get("family", "GC-KS")
    if family == "GC-KS":
        cls, keys = My_CopulaKSVAE_withEigen, KS_PRED_KEYS
    elif family == "GC-TN":
        from MODULES.baseline_GaussCopula_GaussMarginals.GCTN_model import GaussCopulaTNVAE as cls
        keys = cls.PRED_KEYS
    elif family == "GC-GMix":
        from MODULES.variant_GaussCopula_GaussMixMarginals.GCGMix_model import GaussCopulaGMixVAE as cls
        keys = cls.PRED_KEYS
    elif family == "GMM":
        from MODULES.baseline_GaussMixture_K5.GMM_model import GaussMixtureVAE, GaussMixtureDiagVAE
        cls = GaussMixtureDiagVAE if info.get("covariance") == "diag" else GaussMixtureVAE
        keys = cls.PRED_KEYS
    else:
        raise ValueError(f"unknown family {family}")
    data_dir = info["data_dir"]
    pcfg = load_problem_config(data_dir)
    _, K_basis, L_inv = load_baseline_matrices(data_dir)
    n_modes, n_sens = int(pcfg["n_modes"]), len(pcfg["sensor_positions"])
    model = cls(input_dim=int(info["input_dim_enc"]), num_dofs=K_basis.shape[-1], n_modes=n_modes,
                Ke_matrices=K_basis, L_inv=L_inv, n_dims=int(info["n_dims"]),
                num_KS=int(info.get("n_components", info.get("num_KS", 1))), num_samples=int(info["n_samples"]),
                gamma=float(info["gamma"]), mean_f=info["mean_f"], std_f=info["std_f"], lbound=float(info["lbound"]),
                likelihood=info["likelihood"], obs_types=info["obs_types"], sensor_nodes=pcfg.get("sensor_nodes"))
    unit = np.full((2, n_modes, n_sens), 1.0 / np.sqrt(n_sens), np.float32)       # unit-norm placeholder shapes
    _ = model((np.zeros((2, n_modes), np.float32), unit, unit, np.ones((2, int(info["n_dims"])), np.float32)))
    how = load_vae_weights(model, os.path.join(run_folder, "final_model_weights.weights.h5"))
    print(f"  weights of {os.path.basename(run_folder)} ({family}) loaded ({how})")
    return model, keys
