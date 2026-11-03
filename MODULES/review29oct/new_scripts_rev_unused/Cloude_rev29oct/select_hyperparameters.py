# -*- coding: utf-8 -*-
"""
select_hyperparameters.py -- Hyperparameter selection on the VALIDATION set only (R3.2).

Trains one model per combination in GRID and ranks them by the validation
negative ELBO (label-free, the quantity the method minimizes). The validation
-log q(theta* | y) is reported as a second, label-based score. Test sets are
never read here; evaluate the selected run with postprocessing_7D_bridge.py.

Spyder : edit GRID / COMMON, F5.
Console: python select_hyperparameters.py --beta 0.5 1 2 --num_KS 3 [--epochs 400] [--gpu 0]
Result : Output/hyperparameter_selection.csv
"""

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=str, default=None)
if _pre.parse_known_args()[0].gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre.parse_known_args()[0].gpu

import config as C  # noqa: E402

GRID = {"beta": [0.5, 1.0, 2.0], "num_KS": [3]}
COMMON = dict()            # any other TrainConfig field, e.g. dict(epochs=400)


def main(grid=GRID, **common):
    import main_KS_rev29oct as trainer
    rows = []
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        setting = dict(zip(keys, values))
        run_dir = trainer.main(**setting, **common)
        timing = C.load_json(run_dir / "timing.json")
        hist = C.load_json(run_dir / "history.json")
        best = timing["best_epoch"]
        rows.append(dict(run=run_dir.name, **setting, best_epoch=best,
                         val_neg_elbo=timing["best_val_monitor"],
                         val_nll=hist["val_nll"][best] if best >= 0 else float("nan")))
        with open(C.OUTPUT_DIR / "hyperparameter_selection.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    chosen = min(rows, key=lambda r: r["val_neg_elbo"])
    print("Selected on the validation negative ELBO:", chosen)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--beta", type=float, nargs="+", default=GRID["beta"])
    ap.add_argument("--num_KS", type=int, nargs="+", default=GRID["num_KS"])
    ap.add_argument("--epochs", type=int)
    a, _ = ap.parse_known_args()
    extra = dict(COMMON)
    if a.epochs:
        extra["epochs"] = a.epochs
    main({"beta": a.beta, "num_KS": a.num_KS}, **extra)
