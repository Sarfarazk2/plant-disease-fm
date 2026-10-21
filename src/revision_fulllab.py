"""
REVISION R2 - Lab-trained classifiers on ALL PlantVillage training images (not the 200-per-class subset).

Why: reviewers may object that the lab source was subsampled. This trains the frozen-feature
classifier on all 30,978 PlantVillage training images of the 28 shared classes, tests it on all
7,564 lab test images and on both field datasets, and compares it with the subset classifier on
exactly the same test images (paired bootstrap CI + permutation test).

Needs the full PlantVillage features first (LONG: ~4 h for BioCLIP 2, ~1.5 h for DINOv2-B):
    python src/extract_features.py --models bioclip2 dinov2_b14 --splits plantvillage_train plantvillage_test --workers 2
Then (CPU, ~15-60 min per model because C is chosen by 3-fold cross-validation on 31k images):
    python src/revision_fulllab.py
Any other model whose full features exist (e.g. clip_b16 from the earlier robustness check) is included.

Output: results/revision/fulllab/full_vs_subset.csv
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import fewshot as fs
import fm
import stats as S

warnings.filterwarnings("ignore", category=ConvergenceWarning)
OUT = fm.ROOT / "results" / "revision" / "fulllab"
APPLE = S.APPLE
TESTS = [("PlantDoc", "plantdoc_test", "28-way"), ("Plant Pathology", "pp_test", "28-way"),
         ("Plant Pathology", "pp_test", "3-way")]


def predict(clf, X, task, classes):
    if task == "3-way":
        cols = [list(clf.classes_).index(c) for c in APPLE]
        return np.array(APPLE)[clf.predict_proba(X)[:, cols].argmax(1)]
    return clf.predict(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, choices=list(fm.MODELS))
    args = ap.parse_args()
    models = args.models or [k for k in fm.MODELS
                             if (fm.FEATURES / k / "plantvillage_train.npz").exists()
                             and (fm.FEATURES / k / "plantvillage_test.npz").exists()]
    if not models:
        print("No full PlantVillage features found. Run first:\n  python src/extract_features.py "
              "--models bioclip2 dinov2_b14 --splits plantvillage_train plantvillage_test --workers 2")
        return
    print("Models with full PlantVillage features:", ", ".join(models))
    classes, _ = fm.class_list()
    rows = []
    for key in models:
        t0 = time.time()
        Xf, yf = S.feats(key, "plantvillage_train")
        Xs, ys = S.feats(key, "plantvillage_train_sub")
        Xlt, ylt = S.feats(key, "plantvillage_test")
        print(f"\n{fm.MODELS[key]['label']}: {len(yf)} full vs {len(ys)} subset training images")
        c_full = fs.choose_c_cv(Xf, yf)[0]
        c_sub = S.lab_c(key, Xs, ys)
        clf_f, clf_s = fs.logreg(Xf, yf, c_full), fs.logreg(Xs, ys, c_sub)
        for lab, clf, c, n in [("full", clf_f, c_full, len(yf)), ("subset", clf_s, c_sub, len(ys))]:
            a = 100 * np.mean(clf.predict(Xlt) == ylt)
            rows.append({"model": key, "model_name": fm.MODELS[key]["label"], "dataset": "PlantVillage lab test (all)",
                         "task": "28-way", "lab_data": lab, "n_train": n, "C": c, "n_test": len(ylt),
                         "accuracy": a, "ci95_low": np.nan, "ci95_high": np.nan})
        for dname, split, task in TESTS:
            Xt, yt = S.feats(key, split)
            cf = (predict(clf_f, Xt, task, classes) == yt).astype(float)
            cs = (predict(clf_s, Xt, task, classes) == yt).astype(float)
            for lab, c_, cc, n in [("full", c_full, cf, len(yf)), ("subset", c_sub, cs, len(ys))]:
                acc, lo, hi = S.ci(cc)
                rows.append({"model": key, "model_name": fm.MODELS[key]["label"], "dataset": dname, "task": task,
                             "lab_data": lab, "n_train": n, "C": c_, "n_test": len(yt), "accuracy": acc,
                             "ci95_low": lo, "ci95_high": hi})
            d, lo, hi, p = S.paired(cf, cs)
            rows.append({"model": key, "model_name": fm.MODELS[key]["label"], "dataset": dname, "task": task,
                         "lab_data": "full minus subset", "n_train": np.nan, "C": np.nan, "n_test": len(yt),
                         "accuracy": d, "ci95_low": lo, "ci95_high": hi, "p": p})
        print(f"  done in {(time.time() - t0) / 60:.1f} min")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT / "full_vs_subset.csv", index=False)
    print("\nAccuracy (%) - all lab images vs 200-per-class subset, same test images")
    for key in models:
        g = df[df.model == key]
        print(f"  {fm.MODELS[key]['label']}")
        for (dname, task), h in g.groupby(["dataset", "task"], sort=False):
            r = h.set_index("lab_data")
            diff = ""
            if "full minus subset" in r.index:
                q = r.loc["full minus subset"]
                diff = f" | diff {q.accuracy:+.1f} [{q.ci95_low:+.1f}, {q.ci95_high:+.1f}] p={q.p:.3f}"
            print(f"    {dname:<28} {task:<7} full {r.loc['full', 'accuracy']:5.1f} | subset "
                  f"{r.loc['subset', 'accuracy']:5.1f}{diff}")
    print("Saved: results/revision/fulllab/full_vs_subset.csv")


if __name__ == "__main__":
    main()
