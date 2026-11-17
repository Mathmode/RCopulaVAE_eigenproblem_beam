#!/usr/bin/env python3
"""
Post-processing (metrics, benchmark corner plots with the SMC reference posteriors, violins) of ONE test set of an
existing run, e.g. the test dataset 'test_fine_uniform' of a run trained before that split existed.
The predictions Test_predicted_props_<split>.npy must exist (predict_split.py writes them).
Usage (project root):
    python MODULES/review29oct_fixed/postprocess_split.py --run-folder <run> --split test_fine_uniform
"""
import os, sys, argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from MODULES.review29oct_fixed.postprocessing import load_run_info, postprocess_test_set

ap = argparse.ArgumentParser()
ap.add_argument("--run-folder", required=True)
ap.add_argument("--split", default="test_fine_uniform")
ap.add_argument("--n-samples", type=int, default=4000)
ap.add_argument("--n-particles", type=int, default=4000)
ap.add_argument("--workers", type=int, default=None)
a = ap.parse_args()
run = a.run_folder if os.path.isabs(a.run_folder) else os.path.join(ROOT, a.run_folder)
info = load_run_info(run)
postprocess_test_set(run, info, a.split, None, a.n_samples, a.n_particles, a.workers, False)
open(os.path.join(run, f"POSTPROCESSING_{a.split}_DONE.txt"), "w").write(f"{a.split} post-processed\n")
print("done:", run, a.split)
