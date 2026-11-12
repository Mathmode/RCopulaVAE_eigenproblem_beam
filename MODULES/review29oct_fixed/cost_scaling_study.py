#!/usr/bin/env python3
"""
R1.7: computational scaling of the proposed method (GC-LogN encoder + differentiable eigenvalue solver).

Measures the time per training epoch (same data, encoder, batch size, Monte-Carlo samples and training step as the
case study) when the number of latent variables D or the number of finite elements of the decoder grows. The
measurements and the trained models are NOT used for accuracy: the inputs are the 7D training data (5 modes, 5 sensors),
only the decoder (FE mesh and damage zoning) changes, which is what determines the cost.

  (a) latent dimension: 60-element decoder, 5, 10, 20, 30, 60 deck zones + 2 bearings -> D = 7, 12, 22, 32, 62
  (b) FE discretization: 5 deck zones + 2 bearings (D = 7), 10, 20, 50, 100 elements -> 22, 42, 102, 202 DOFs

Fits t_epoch = a * x^b (log-log least squares) in each study and writes Output/cost_scaling/cost_scaling.csv/.json.
Usage (project root): python MODULES/review29oct_fixed/cost_scaling_study.py [--gpu 0] [--epochs 4]
"""
import os, sys, json, time, argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=str, default=None)
ap.add_argument("--epochs", type=int, default=4, help="epochs per configuration (the first one is discarded)")
ap.add_argument("--out", type=str, default=os.path.join(PROJECT_ROOT, "Output", "cost_scaling"))
args = ap.parse_args()
from MODULES.review29oct_fixed.training_common import select_gpu
select_gpu(args.gpu)

import tensorflow as tf
from tensorflow import keras as K
from MODULES.review29oct_fixed import config as C
from MODULES.review29oct_fixed.fe_model import BeamMesh
from MODULES.review29oct_fixed.preprocessing_and_eigen import load_bridge_data
from MODULES.review29oct_fixed.reference_posterior import load_problem_config
from MODULES.baseline_GaussCopula_GaussMarginals.GCLogN_model import GaussCopulaLogNVAE

DATA = os.path.join(PROJECT_ROOT, "Data", "06Nov_FINAL_SlabS10_7D_5sensors_5modes")
pcfg = load_problem_config(DATA)
lik = pcfg["likelihood"]
(f_tr, r_tr, v_tr, a_tr, *_rest, mean_f, std_f) = load_bridge_data(DATA)
os.makedirs(args.out, exist_ok=True)


class EpochTimer(K.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.t0)


def time_config(n_zones, n_el):
    mesh = BeamMesh(n_zones, n_el // n_zones, sensor_positions=C.SENSOR_POSITIONS, **C.STRUCTURE)
    K_basis, M = mesh.zone_basis()
    L_inv = np.linalg.inv(np.linalg.cholesky(M))
    D = n_zones + 2
    a = np.ones((len(f_tr), D), np.float32)                     # labels are not used by the training step
    ds = tf.data.Dataset.from_tensor_slices(((f_tr, r_tr, v_tr, a), a)).cache().shuffle(len(f_tr), seed=0).batch(256)
    tf.keras.utils.set_random_seed(0)
    model = GaussCopulaLogNVAE(input_dim=f_tr.shape[1] + v_tr.shape[1] * v_tr.shape[2], num_dofs=K_basis.shape[-1],
                               n_modes=f_tr.shape[1], Ke_matrices=K_basis.astype(np.float32),
                               L_inv=L_inv.astype(np.float32), n_dims=D, num_KS=1, num_samples=6, gamma=1.0,
                               mean_f=mean_f, std_f=std_f, lbound=C.LBOUND, likelihood=lik, obs_types="fv",
                               sensor_nodes=mesh.sensor_nodes.tolist())
    model.compile(optimizer=K.optimizers.Adam(5e-4, global_clipnorm=10.0))
    _ = model((f_tr[:2], r_tr[:2], v_tr[:2], a[:2]))
    n_par = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    cb = EpochTimer()
    model.fit(ds, epochs=args.epochs, verbose=0, callbacks=[cb])
    t = float(np.median(cb.times[1:]))
    row = dict(D=D, n_zones=n_zones, n_elements=n_el, n_dofs=int(K_basis.shape[-1]), posterior_params=D * (D + 3) // 2,
               encoder_params=n_par, time_per_epoch_s=t, epoch_times=cb.times)
    print(json.dumps({k: v for k, v in row.items() if k != "epoch_times"}), flush=True)
    return row


def fit(x, y):
    b, la = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(la)), float(b)


rows = {"latent": [time_config(z, 60) for z in (5, 10, 20, 30, 60)],
        "mesh": [time_config(5, n) for n in (10, 20, 50, 100)]}
res = {"rows": rows,
       "fit_time_vs_D": fit([r["D"] for r in rows["latent"]], [r["time_per_epoch_s"] for r in rows["latent"]]),
       "fit_time_vs_dofs": fit([r["n_dofs"] for r in rows["mesh"]], [r["time_per_epoch_s"] for r in rows["mesh"]]),
       "device": tf.config.list_physical_devices("GPU")[0].name if tf.config.list_physical_devices("GPU") else "CPU"}
print("t_epoch ~ a * D^b    :", res["fit_time_vs_D"])
print("t_epoch ~ a * DOFs^b :", res["fit_time_vs_dofs"])
with open(os.path.join(args.out, "cost_scaling.json"), "w") as fh:
    json.dump(res, fh, indent=2)
import csv
with open(os.path.join(args.out, "cost_scaling.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["study", "D", "n_elements", "n_dofs", "posterior_params", "encoder_params", "time_per_epoch_s"])
    for k, rr in rows.items():
        for r in rr:
            w.writerow([k, r["D"], r["n_elements"], r["n_dofs"], r["posterior_params"], r["encoder_params"],
                        f"{r['time_per_epoch_s']:.4f}"])
