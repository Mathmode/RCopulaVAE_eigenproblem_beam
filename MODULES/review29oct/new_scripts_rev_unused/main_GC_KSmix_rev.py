#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULES/review29oct/new_scripts_rev/main_GC_KSmix_rev.py

Training script for the mixture-of-Gaussian-copulas / Kumaraswamy-marginal PI-VAE
on the 7-parameter bridge with elastic bearings.

Run from the repository root, e.g.
    python MODULES/review29oct/new_scripts_rev/main_GC_KSmix_rev.py --exp Exp1 --epochs 2000 --gpu 2
Data are read from Data/<exp>_BridgeBearings_7D and results written to Output/<exp>/<run_name>/.
Quick smoke test:
    python MODULES/review29oct/new_scripts_rev/main_GC_KSmix_rev.py --epochs 3 --n_train 2048 --no_eval
"""

import os
import sys
import json
import time
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--exp', type=str, default='Exp1',
                    help='experiment tag: data from Data/<exp>_BridgeBearings_7D, results to Output/<exp>/')
parser.add_argument('--data_dir', type=str, default=None, help='overrides the data folder given by --exp')
parser.add_argument('--out_dir', type=str, default=None, help='overrides the results folder given by --exp')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--n_comp', type=int, default=3, help='mixture components (1 = single Gaussian copula)')
parser.add_argument('--num_samples', type=int, default=4, help='MC samples PER component per observation')
parser.add_argument('--gamma', type=float, default=0.25)
parser.add_argument('--prior_alpha', type=float, default=2.2)
parser.add_argument('--prior_beta', type=float, default=1.1)
parser.add_argument('--objective', type=str, default='elbo', choices=['elbo', 'siwae'])
parser.add_argument('--kl_anneal_start', type=float, default=3.0, help='1.0 disables annealing')
parser.add_argument('--kl_anneal_epochs', type=int, default=200)
parser.add_argument('--patience', type=int, default=300)
parser.add_argument('--n_train', type=int, default=None, help='subsample training set (debugging)')
parser.add_argument('--no_eval', action='store_true')
args = parser.parse_args()
args.data_dir = args.data_dir or os.path.join(ROOT, 'Data', f'{args.exp}_BridgeBearings_7D')
args.out_dir = args.out_dir or os.path.join(ROOT, 'Output', args.exp)

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['KERAS_BACKEND'] = 'tensorflow'   # custom train_step below is TF-specific

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from MODULES.review29oct.preprocessing_and_eigen import load_bridge_data, load_baseline_matrices
from MODULES.review29oct.new_scripts_rev.GC_KSmix_models import MixGC_KS_VAE_withEigen, KLWeightAnnealing


class BestWeightsAfterAnnealing(K.callbacks.Callback):
    """Keeps the best val_loss weights in RAM, only once the exact objective (kl_weight = 1) is active."""
    def __init__(self, start_epoch, patience):
        super().__init__()
        self.start_epoch, self.patience = start_epoch, patience
        self.best, self.best_epoch, self.best_weights, self.wait = np.inf, 0, None, 0

    def on_epoch_end(self, epoch, logs=None):
        val = (logs or {}).get('val_loss')
        if val is None or epoch < self.start_epoch:
            return
        if val < self.best:
            self.best, self.best_epoch, self.wait = val, epoch + 1, 0
            self.best_weights = [w.numpy() for w in self.model.weights]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            for var, val in zip(self.model.weights, self.best_weights):
                var.assign(val)
            print(f"Restored best weights from epoch {self.best_epoch} (val_loss = {self.best:.4f})")


def build_model(cfg, K_basis, L_inv, mean_f, std_f):
    return MixGC_KS_VAE_withEigen(
        input_dim=cfg['input_dim'], num_dofs=cfg['n_dofs'], n_modes=cfg['n_modes'],
        Ke_matrices=K_basis, L_inv=L_inv, n_dims=cfg['n_dims'], n_comp=cfg['n_comp'],
        num_samples=cfg['num_samples'], gamma=cfg['gamma'], mean_f=mean_f, std_f=std_f,
        lbound=cfg['lbound'], prior_alpha=cfg['prior_alpha'], prior_beta=cfg['prior_beta'],
        lambda_vert=cfg['lambda_vert'], lambda_rot=cfg['lambda_rot'], objective=cfg['objective'])


def main():
    K.utils.set_random_seed(1234)
    K.backend.set_floatx('float32')

    (f_train, r_train, v_train, a_train,
     f_val, r_val, v_val, a_val,
     f_test, r_test, v_test, a_test,
     mean_f, std_f) = load_bridge_data(args.data_dir)
    Mfree, K_basis, L_inv = load_baseline_matrices(args.data_dir)
    if args.n_train is not None:
        f_train, r_train, v_train, a_train = (arr[:args.n_train] for arr in (f_train, r_train, v_train, a_train))

    n_modes = f_train.shape[1]
    cfg = {
        'exp': args.exp, 'data_dir': args.data_dir, 'n_dims': int(K_basis.shape[0]), 'n_dofs': int(K_basis.shape[1]),
        'n_modes': int(n_modes), 'input_dim': int(n_modes + r_train[0].size + v_train[0].size),
        'lbound': 0.45, 'n_comp': args.n_comp, 'num_samples': args.num_samples, 'gamma': args.gamma,
        'prior_alpha': args.prior_alpha, 'prior_beta': args.prior_beta,
        'lambda_vert': 1.5, 'lambda_rot': 0.5, 'objective': args.objective,
        'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
        'kl_anneal_start': args.kl_anneal_start, 'kl_anneal_epochs': args.kl_anneal_epochs,
    }
    run_name = (f"{args.exp}_MixGC{cfg['n_comp']}_KS_{cfg['objective']}_Gamma{cfg['gamma']}"
                f"_S{cfg['num_samples']}_Ep{cfg['epochs']}")
    folder = os.path.join(args.out_dir, run_name)
    os.makedirs(folder, exist_ok=True)

    train_ds = (tf.data.Dataset.from_tensor_slices(((f_train, r_train, v_train, a_train), a_train))
                .cache().shuffle(len(f_train), seed=1234).batch(args.batch_size).prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices(((f_val, r_val, v_val, a_val), a_val))
              .cache().batch(args.batch_size).prefetch(tf.data.AUTOTUNE))

    model = build_model(cfg, K_basis, L_inv, mean_f, std_f)
    steps = int(np.ceil(len(f_train) / args.batch_size))
    lr = K.optimizers.schedules.CosineDecay(args.lr, decay_steps=args.epochs * steps, alpha=0.02)
    model.compile(optimizer=K.optimizers.Adam(learning_rate=lr, clipnorm=1.0), jit_compile=False)
    _ = model((f_train[:2], r_train[:2], v_train[:2], a_train[:2]))

    anneal_epochs = args.kl_anneal_epochs if args.kl_anneal_start != 1.0 else 0
    callbacks = [BestWeightsAfterAnnealing(start_epoch=anneal_epochs, patience=args.patience),
                 K.callbacks.TerminateOnNaN()]
    if anneal_epochs > 0:
        callbacks.insert(0, KLWeightAnnealing(args.kl_anneal_start, anneal_epochs))

    print(f"Training {run_name} on {len(f_train)} samples ...")
    t0 = time.time()
    history = model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=callbacks, verbose=2)
    cfg['training_time_sec'] = time.time() - t0
    print(f"Training time: {cfg['training_time_sec']:.1f} s")

    model.save_weights(os.path.join(folder, 'final_model_weights.weights.h5'))
    np.save(os.path.join(folder, 'model_history.npy'), history.history, allow_pickle=True)
    with open(os.path.join(folder, 'config.json'), 'w') as fh:
        json.dump(cfg, fh, indent=2)
    print(f"Saved weights, history and config to {folder}")

    if not args.no_eval:
        from MODULES.review29oct.new_scripts_rev.evaluate_posteriors import run_evaluation
        run_evaluation(folder, model=model)


if __name__ == '__main__':
    main()
