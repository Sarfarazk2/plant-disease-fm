"""
Step 6a - Fixed, class-balanced subsets of the PlantVillage (lab) images.
Running time: a few seconds.

Why: a linear classifier on foundation-model features saturates with a few hundred lab
images per class, while extracting features for all ~38k lab images takes hours per model
on a 4 GB laptop GPU. All models therefore use the same fixed subset; CLIP is additionally
run on the FULL lab set to show that the subset gives the same conclusions.

Run from the project folder (plant-disease-fm):
    python src/make_lab_subsets.py

Output:
    data/splits/plantvillage_train_sub.csv   200 images per shared class (all if fewer)
    data/splits/plantvillage_test_sub.csv    100 images per shared class (all if fewer)
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
PER_CLASS = {"plantvillage_train": 200, "plantvillage_test": 100}
SEED = 0


def main():
    for split, n in PER_CLASS.items():
        with open(SPLITS / f"{split}.csv", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["shared"] == "1"]
        by_class = defaultdict(list)
        for r in rows:
            by_class[r["label"]].append(r)

        rng = np.random.default_rng(SEED)
        out = []
        for lab in sorted(by_class):
            pool = sorted(by_class[lab], key=lambda r: r["path"])
            take = min(n, len(pool))
            out += [pool[i] for i in sorted(rng.choice(len(pool), size=take, replace=False))]

        path = SPLITS / f"{split}_sub.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(out)
        smallest = min(len(v) for v in by_class.values())
        print(f"{path.name}: {len(out)} images, {len(by_class)} classes "
              f"(up to {n} per class; smallest class has {smallest} available)")


if __name__ == "__main__":
    main()
