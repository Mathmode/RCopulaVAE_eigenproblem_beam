#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/benchmark_train_step.py

Times the pieces of one training step of the 7D Copula-KS VAE, so we can see where the
time goes on a given machine (run it on the remote GPU). It does NOT train or save anything.

Run from the project root:
    python MODULES/review29oct/benchmark_train_step.py
"""

import os
import sys
import time
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import tensorflow as tf
import tensorflow.keras as K

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from MODULES.review29oct.preprocessing_and_eigen import load_bridge_data, load_baseline_matrices
from MODULES.review29oct.GC_KSmix_models import My_CopulaKSVAE_withEigen

DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "V01_BridgeBearings_7D_DoE")
BATCH_SIZE = 256
NUM_SAMPLES = 6
N_REPEATS = 10


def timeit(fn, n=N_REPEATS):
    out = fn()  # warm-up / tracing / XLA compile
    _ = [np.asarray(t) for t in tf.nest.flatten(out)]
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    _ = [np.asarray(t) for t in tf.nest.flatten(out)]  # forces device sync
    return (time.perf_counter() - t0) / n


def build_model(data, K_basis, L_inv, jit_compile, eig_device=None):
    (f_tr, r_tr, v_tr, a_tr, *_rest, mean_f, std_f) = data
    input_dim = f_tr.shape[1] + r_tr.shape[1] * r_tr.shape[2] + v_tr.shape[1] * v_tr.shape[2]
    n_dofs = K_basis.shape[-1]
    kwargs = {} if eig_device is None else {"eig_device": eig_device}
    model = My_CopulaKSVAE_withEigen(
        input_dim=input_dim, num_dofs=n_dofs, n_modes=f_tr.shape[1], K_basis=K_basis,
        L_inv=L_inv, n_dims=a_tr.shape[1], num_KS=3, num_samples=NUM_SAMPLES, gamma=0.35,
        mean_f=mean_f, std_f=std_f, lbound=0.45, **kwargs,
    )
    _ = model((f_tr[:1], r_tr[:1], v_tr[:1], a_tr[:1]))
    model.compile(
        optimizer=K.optimizers.Adam(1e-4, clipnorm=0.5),
        loss=model.ELBO_Copula_loss,
        metrics=[model.Freqs_loss, model.MAC_modes_loss, model.KL_Divergence_term],
        jit_compile=jit_compile,
    )
    return model


def main():
    K.utils.set_random_seed(1234)
    K.backend.set_floatx("float32")
    print("TF", tf.__version__, "| devices:", [d.name for d in tf.config.list_physical_devices()])
    data = load_bridge_data(DATA_PATH)
    _, K_basis, L_inv = load_baseline_matrices(DATA_PATH)
    f_tr, r_tr, v_tr, a_tr = data[0], data[1], data[2], data[3]
    xb = tuple(tf.constant(t[:BATCH_SIZE]) for t in (f_tr, r_tr, v_tr, a_tr))
    yb = xb[3]
    steps_per_epoch = int(np.ceil(len(f_tr) / BATCH_SIZE))
    n_mat = BATCH_SIZE * NUM_SAMPLES
    n_dofs = K_basis.shape[-1]

    # --- 1. Eigensolver alone: (BATCH*S) symmetric n_dofs x n_dofs matrices ---
    A = np.random.rand(n_mat, n_dofs, n_dofs).astype(np.float32)
    A = tf.constant(A @ np.transpose(A, (0, 2, 1)) + n_dofs * np.eye(n_dofs, dtype=np.float32))

    def eigh_grad(device, jit):
        @tf.function(jit_compile=jit)
        def f(A):
            with tf.GradientTape() as tape:
                tape.watch(A)
                with tf.device(device):
                    w, v = tf.linalg.eigh(A)
                loss = tf.reduce_sum(w[:, :5]) + tf.reduce_sum(tf.square(v[:, :, :5]))
            return tape.gradient(loss, A)
        return lambda: f(A)

    print(f"\n[1] eigh fwd+bwd on {n_mat} matrices of {n_dofs}x{n_dofs}")
    devices = ["/CPU:0"] + (["/GPU:0"] if tf.config.list_physical_devices("GPU") else [])
    for dev in devices:
        for jit in (False, True):
            try:
                print(f"    device={dev:7s} jit={jit!s:5s}: {1e3 * timeit(eigh_grad(dev, jit)):8.1f} ms")
            except Exception as e:
                print(f"    device={dev:7s} jit={jit!s:5s}: FAILED ({type(e).__name__})")

    # --- 2. Full Keras train step under different settings ---
    print(f"\n[2] full train_step (batch={BATCH_SIZE}, samples={NUM_SAMPLES}); epoch = {steps_per_epoch} steps")
    configs = [("original (jit auto)", "auto", None),
               ("jit_compile=True", True, None),
               ("jit_compile=False", False, None),
               ("jit_compile=False + eigh on CPU", False, "/CPU:0")]
    for name, jit, eig_dev in configs:
        try:
            model = build_model(data, K_basis, L_inv, jit, eig_dev)
        except TypeError:
            print(f"    {name:34s}: skipped (model has no eig_device option)")
            continue
        try:
            model.make_train_function()
            print(f"    ({name}: model.jit_compile={model.jit_compile})")
            it = iter(tf.data.Dataset.from_tensors((xb, yb)).repeat())
            step = lambda: model.train_function(it)
            dt = timeit(step)
            print(f"    {name:34s}: {1e3 * dt:8.1f} ms/step  ->  ~{dt * steps_per_epoch:6.1f} s/epoch (train only)")
        except Exception as e:
            print(f"    {name:34s}: FAILED ({type(e).__name__}: {str(e)[:150]})")


if __name__ == "__main__":
    main()
