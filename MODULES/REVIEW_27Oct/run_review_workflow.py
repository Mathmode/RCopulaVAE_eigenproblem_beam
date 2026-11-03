# -*- coding: utf-8 -*-
"""
run_review_workflow.py -- Master Orchestrator for the MS CPENG-8674 Revision.

Guarantees that ALL datasets and ALL results/models/tables generated for the
review are stored in dedicated, self-documented folders:
  - Datasets : `code/Data/review_27oct/<DATASET_NAME>/`
  - Outputs  : `code/Output/review_27oct/<WORK_PACKAGE>/<EXP_NAME>/`

Usage (from `code/` directory):
  python -m MODULES.review_27oct.run_review_workflow --status
  python -m MODULES.review_27oct.run_review_workflow --wp WP1 --epochs 2000
  python -m MODULES.review_27oct.run_review_workflow --wp WP2 --epochs 2000
  python -m MODULES.review_27oct.run_review_workflow --wp WP3_WP4_eval
"""

from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np

from .generate_review_datasets import REGISTRY, generate
from .review_uq_and_training_utils import (
    exact_truncated_normal_ci,
    approximate_clipped_ci,
    compute_coverage_and_width,
    evaluate_healthy_false_positive_rate,
    select_gamma_on_validation,
)

DATA_ROOT = os.path.join("Data", "review_27oct")
OUTPUT_ROOT = os.path.join("Output", "review_27oct")


WORK_PACKAGES = {
    "WP1": {
        "title": "Calibration, Validation-Set Gamma Selection, H-Ablation & Runtime",
        "comments": ["Editor (2,3)", "R1.3", "R3.2", "R3.3", "R3.7", "R3.8"],
        "datasets": ["TRAIN_E5_uniform", "TEST_E5_iid", "TEST_E5_infprior"],
        "output_dir": os.path.join(OUTPUT_ROOT, "WP1_Calibration_and_Validation"),
    },
    "WP2": {
        "title": "Quantitative Baseline Comparisons (PI-GCVAE vs Surrogate-NN, TMVN, GMM-VAE, KSmix)",
        "comments": ["Editor (4)", "R1.4"],
        "datasets": ["TRAIN_E5_uniform", "TEST_E5_iid"],
        "output_dir": os.path.join(OUTPUT_ROOT, "WP2_Baselines_Comparison"),
    },
    "WP3": {
        "title": "Healthy False-Positive Rate (FPR), Localized Mild Damage & OOD Stress Tests",
        "comments": ["Editor (1)", "R1.5", "R1.6", "R3.3"],
        "datasets": [
            "TRAIN_E5_spikeslab",
            "TEST_E5_healthy",
            "TEST_E5_single",
            "TEST_E5_double",
            "TEST_E5_oodsevere",
            "TEST_E5_noise2x",
            "TEST_E5_noise4x",
        ],
        "output_dir": os.path.join(OUTPUT_ROOT, "WP3_Localized_Healthy_and_OOD"),
    },
    "WP4": {
        "title": "Model-Form Uncertainty: Fine FE Meshes (20/50/100), Mass Perturbations & Mode Count n_m",
        "comments": ["Editor (1)", "R1.1", "R1.2", "R3.1", "R3.4"],
        "datasets": [
            "06Oct_Data_Noisy_E5_Lvl25_MassPerturb5",
            "TRAIN_E5_massnuis",
            "TEST_E5_finemesh20",
            "TEST_E5_finemesh50",
            "TEST_E5_finemesh100",
            "TEST_E5_massR010",
            "TEST_E5_massR025",
            "TEST_E5_massR050",
            "TEST_E5_massR100",
            "TEST_E5_massG050",
            "TEST_E5_eov",
        ],
        "output_dir": os.path.join(OUTPUT_ROOT, "WP4_ModelForm_Mesh_Mass_Modes"),
    },
    "WP5": {
        "title": "Realistic Short-Span Bridge, Elastic Bearings, Sub-Zone Cracks & Scalability",
        "comments": ["Editor (1)", "R1.7", "R2.1", "R2.2", "R2.3", "R2.4"],
        "datasets": [
            "TRAIN_toy_elastic",
            "TEST_E5_elasticdata",
            "TEST_toy_elastic_iid",
            "TEST_toy_elastic_supportonly",
            "TEST_E5_localized_fine50",
            "TRAIN_bridge20",
            "TEST_bridge20_iid",
            "TEST_bridge20_healthy",
            "TEST_bridge20_single",
            "TEST_bridge20_localized",
            "TEST_bridge20_mass05",
            "TRAIN_scal_Z5",
            "TRAIN_scal_Z10",
            "TRAIN_scal_Z20",
            "TRAIN_scal_Z40",
        ],
        "output_dir": os.path.join(OUTPUT_ROOT, "WP5_Realistic_Bridge_Bearings_Scaling"),
    },
}


def load_review_dataset(dataset_name: str, data_root: str = DATA_ROOT, n_modes: int | None = None):
    """
    Load a dataset from `code/Data/review_27oct/<dataset_name>/` using its deterministic
    `split_indices.npz` (if a training set) or as a full evaluation set (if a test set).
    """
    path = os.path.join(data_root, dataset_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset {dataset_name} not found at {path}. "
            f"Run `python -m MODULES.review_27oct.generate_review_datasets --only {dataset_name}`."
        )

    F = np.load(os.path.join(path, "freqs_data_true.npy")).astype(np.float32)
    R = np.load(os.path.join(path, "rotmodes_data_true.npy")).astype(np.float32)
    V = np.load(os.path.join(path, "vertmodes_data_true.npy")).astype(np.float32)
    Z = np.load(os.path.join(path, "alpha_factors_true.npy")).astype(np.float32)

    if n_modes is not None:
        F = F[:, :n_modes]
        R = R[:, :n_modes, :]
        V = V[:, :n_modes, :]

    Mfree = np.load(os.path.join(path, "Mass_matrix.npy")).astype(np.float32)
    Ke = np.load(os.path.join(path, "Ke_matrix.npy")).astype(np.float32)
    L_inv = np.load(os.path.join(path, "L_inv.npy")).astype(np.float32)

    split_file = os.path.join(path, "split_indices.npz")
    if os.path.exists(split_file):
        sp = np.load(split_file)
        tr, va, te = sp["train"], sp["val"], sp["test"]
        F_log_tr = np.log(np.maximum(F[tr], 1e-7))
        mean_f = np.mean(F_log_tr, axis=0)
        std_f = np.std(F_log_tr, axis=0) + 1e-8

        def _scale_f(f_arr):
            return ((np.log(np.maximum(f_arr, 1e-7)) - mean_f) / std_f).astype(np.float32)

        return {
            "train": (_scale_f(F[tr]), R[tr], V[tr], Z[tr]),
            "val": (_scale_f(F[va]), R[va], V[va], Z[va]),
            "test": (_scale_f(F[te]), R[te], V[te], Z[te]),
            "raw_freqs": {"train": F[tr], "val": F[va], "test": F[te]},
            "mean_f": mean_f.astype(np.float32),
            "std_f": std_f.astype(np.float32),
            "Mfree": Mfree,
            "Ke": Ke,
            "L_inv": L_inv,
            "path": path,
        }
    else:
        return {
            "F_raw": F,
            "R": R,
            "V": V,
            "Z": Z,
            "Mfree": Mfree,
            "Ke": Ke,
            "L_inv": L_inv,
            "path": path,
        }


def print_status():
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    status_lines = [
        "# MS CPENG-8674 Revision Workflow Status",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "- **Datasets Root**: `code/Data/review_27oct/`",
        "- **Outputs Root**: `code/Output/review_27oct/`\n",
    ]
    print("\n" + "=" * 78)
    print("MS CPENG-8674 REVISION WORKFLOW STATUS")
    print(f"  Datasets folder : code/{DATA_ROOT}")
    print(f"  Outputs folder  : code/{OUTPUT_ROOT}")
    print("=" * 78)

    for wp_id, wp in WORK_PACKAGES.items():
        os.makedirs(wp["output_dir"], exist_ok=True)
        ds_done = [d for d in wp["datasets"] if os.path.exists(os.path.join(DATA_ROOT, d, "metadata.json"))]
        print(f"\n[{wp_id}] {wp['title']}")
        print(f"  Addresses : {', '.join(wp['comments'])}")
        print(f"  Output dir: code/{wp['output_dir']}")
        print(f"  Datasets  : {len(ds_done)}/{len(wp['datasets'])} ready in code/{DATA_ROOT}/")
        for d in wp["datasets"]:
            mark = "READY" if d in ds_done else "PENDING"
            print(f"    - [{mark:7s}] {d}")

        status_lines.append(f"## {wp_id}: {wp['title']}")
        status_lines.append(f"- **Reviewer Comments**: {', '.join(wp['comments'])}")
        status_lines.append(f"- **Output Directory**: `{wp['output_dir']}`")
        status_lines.append(f"- **Datasets ({len(ds_done)}/{len(wp['datasets'])} generated)**:")
        for d in wp["datasets"]:
            mark = "x" if d in ds_done else " "
            status_lines.append(f"  - [{mark}] `Data/review_27oct/{d}` — {REGISTRY[d]['comment']}")
        status_lines.append("")

    status_file = os.path.join(OUTPUT_ROOT, "WORKFLOW_STATUS.md")
    with open(status_file, "w", encoding="utf-8") as f:
        f.write("\n".join(status_lines))
    print(f"\nSaved workflow tracker to code/{status_file}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--status", action="store_true", help="Print and save workflow status across WP1-WP5")
    p.add_argument("--ensure-datasets", choices=list(WORK_PACKAGES.keys()) + ["all"], default=None,
                   help="Generate any missing datasets for the specified Work Package")
    args = p.parse_args()

    if args.ensure_datasets:
        wps = list(WORK_PACKAGES.keys()) if args.ensure_datasets == "all" else [args.ensure_datasets]
        for wp_id in wps:
            for d in WORK_PACKAGES[wp_id]["datasets"]:
                if not os.path.exists(os.path.join(DATA_ROOT, d, "metadata.json")):
                    generate(REGISTRY[d], DATA_ROOT)
    print_status()


if __name__ == "__main__":
    main()
