#!/usr/bin/env python3
"""
Encoder outputs of a trained run (any family handled by training_common.load_trained_model) for a test split of its
data set, written as <run>/Test_predicted_props_<split>.npy (the name used by the training scripts), e.g. for the
split 'test_fine_uniform' added after the training (add_test_fine_uniform.py).
Usage (project root): python MODULES/review29oct_fixed/predict_split.py --run <run folder> --split test_fine_uniform
"""
import os, re, sys, argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--split", default="test_fine_uniform")
a = ap.parse_args()
run = a.run if os.path.isabs(a.run) else os.path.join(ROOT, a.run)
info = np.load(os.path.join(run, "Problem_info.npy"), allow_pickle=True).item()
if not os.path.isdir(info["data_dir"]):
    info["data_dir"] = os.path.join(ROOT, "Data", re.split(r"[\\/]", str(info["data_dir"]).rstrip("\\/"))[-1])
from MODULES.review29oct_fixed.training_common import load_trained_model
from MODULES.review29oct_fixed.preprocessing_and_eigen import load_test_split
model, keys = load_trained_model(run, info)
f, r, v, z = load_test_split(info["data_dir"], a.split, info["mean_f"], info["std_f"])
outs = model.Encoder_model.predict((f, r.astype(np.float32), v.astype(np.float32), z.astype(np.float32)), batch_size=256,
                                   verbose=0)
out = os.path.join(run, f"Test_predicted_props_{a.split}.npy")
np.save(out, dict(zip(keys, outs)), allow_pickle=True)
print(f"[OK] {out}")
