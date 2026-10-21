"""
REVISION R3 - Stronger few-shot baselines + a new question: do lab images still help once a few
field images are available?  (CPU only, uses saved features; about 45-90 minutes.)

Part A - Tip-Adapter (Zhang et al., ECCV 2022), the standard training-free few-shot adapter for
vision-language models, compared with our reliability-weighted fusion (v2) on the SAME support sets:
    Tip-Adapter (default)   alpha = 1, beta = 5.5 (values recommended by the authors)
    Tip-Adapter (support)   alpha, beta chosen on the K support images by leave-one-out
                            (no field test data, same rule as v2; K = 1 -> defaults)
  5 vision-language models x 3 field datasets x K = 1, 5, 10, 16 x 5 support sets = the 60 settings
  of the paper's fusion table.

Part B - Lab + field training (all 10 encoders; PlantDoc 28-way and apple 3-way; wheat has no lab
classes). Linear classifier trained on
    field only              the K support images (= the paper's few-shot linear probe)
    lab only                the 200-per-class lab subset (= the paper's lab-trained classifier)
    lab + field             lab subset + K support images pooled
    lab + field (balanced)  as above, but the support images get the same total weight as the lab images
  C = the lab classifier's C (chosen on lab data only).

Run from the project folder (plant-disease-fm):
    python src/revision_baselines.py
    python src/revision_baselines.py --parts A        # only Tip-Adapter

Output: results/revision/baselines/tip_runs.csv, tip_vs_v2.csv,
        results/revision/baselines/labfield_runs.csv, labfield_summary.csv
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import fewshot as fs
import fm
import fusion as fu
import stats as S

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

OUT = fm.ROOT / "results" / "revision" / "baselines"
TIP_ALPHA, TIP_BETA = 1.0, 5.5
ALPHA_GRID = [0.1, 0.5, 1, 2, 5, 10]
BETA_GRID = [1, 3, 5.5, 7, 10]
DATASETS = {
    "plantdoc": {"test": "plantdoc_test", "pool": "plantdoc_train_clean", "fewshot": "fewshot_k{k}_seed{s}"},
    "pp": {"test": "pp_test", "pool": "pp_pool", "fewshot": "pp_fewshot_k{k}_seed{s}"},
    "wheat": {"test": "wheat_test", "pool": "wheat_pool", "fewshot": "wheat_fewshot_k{k}_seed{s}"},
}
VLMS = [k for k in ["clip_b16", "bioclip", "bioclip2", "clip_l14", "siglip_b16"] if k in fm.MODELS]


# ------------------------------------------------------------------ shared data loading
def load(key, dname):
    cfg = DATASETS[dname]
    test, pool = fs.load_feats(key, cfg["test"]), fs.load_feats(key, cfg["pool"])
    if test is None or pool is None:
        return None
    all28, _ = fm.class_list()
    if dname == "plantdoc":
        classes = all28
    elif dname == "pp":
        classes = fu.APPLE
    else:
        classes = sorted(set(test[1]))
    return test, pool, classes, all28


def text_w(key, dname, classes, all28):
    if not fm.has_text_encoder(key):
        return None
    if dname == "wheat":
        import eval_wheat
        return eval_wheat.wheat_text(key, classes, fm.MODELS[key]["prompt"])
    return fs.text_weights(key, all28)[[all28.index(c) for c in classes]]


def support(cfg, k, s, index, Xp, yp):
    sup = fs.read_csv(fm.SPLITS / (cfg["fewshot"].format(k=k, s=s) + ".csv"))
    idx = [index[r["path"]] for r in sup if r["path"] in index]
    return Xp[idx], yp[idx]


def lab_data(key, classes):
    tr, te = fs.load_feats(key, "plantvillage_train_sub"), fs.load_feats(key, "plantvillage_test_sub")
    m1, m2 = np.isin(tr[1], classes), np.isin(te[1], classes)
    return (tr[0][m1], tr[1][m1], None), (te[0][m2], te[1][m2], None)


# ------------------------------------------------------------------ Part A: Tip-Adapter
def tip_logits(Xq, Xs, L, tw, alpha, beta, self_mask=False):
    A = np.exp(-beta * (1.0 - Xq @ Xs.T))
    if self_mask:                      # leave-one-out on the support set: an image never votes for itself
        np.fill_diagonal(A, 0.0)
    return 100.0 * Xq @ tw.T + alpha * A @ L


def tip_support_params(Xs, ys, tw, classes):
    counts = pd.Series(ys).value_counts()
    if counts.max() < 2:
        return TIP_ALPHA, TIP_BETA, False
    L = np.eye(len(classes))[[classes.index(y) for y in ys]]
    truth = L.argmax(1)
    best, best_acc = (TIP_ALPHA, TIP_BETA), -1.0
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            acc = np.mean(tip_logits(Xs, Xs, L, tw, a, b, self_mask=True).argmax(1) == truth)
            # ties -> closest to the authors' defaults
            if acc > best_acc + 1e-12 or (abs(acc - best_acc) < 1e-12 and
                                          abs(np.log(a)) + abs(b - TIP_BETA) / 5.5 <
                                          abs(np.log(best[0])) + abs(best[1] - TIP_BETA) / 5.5):
                best, best_acc = (a, b), acc
    return best[0], best[1], True


def part_a():
    rows = []
    for key in VLMS:
        for dname, cfg in DATASETS.items():
            d = load(key, dname)
            if d is None:
                print(f"  {key} / {dname}: features missing - skipped")
                continue
            (Xt, yt, _), (Xp, yp, pp), classes, all28 = d
            cls = np.array(classes)
            index = {p: i for i, p in enumerate(pp)}
            tw = text_w(key, dname, classes, all28)
            zt = fu.z(Xt @ tw.T)
            lab_tr, lab_te = lab_data(key, all28 if dname == "wheat" else classes)
            t0 = time.time()
            for k in fs.SHOTS:
                c_k = fs.choose_c_fewshot(lab_tr, lab_te, all28 if dname == "wheat" else classes, k)[0]
                for s in fs.SEEDS:
                    Xs, ys = support(cfg, k, s, index, Xp, yp)
                    L = np.eye(len(classes))[[classes.index(y) for y in ys]]

                    def add(method, score, **extra):
                        rows.append({"dataset": dname, "model": key, "method": method, "K": k, "seed": s,
                                     "accuracy": 100 * np.mean(cls[score.argmax(1)] == yt), **extra})

                    add("text only", zt)
                    zi = fu.z(fu.class_logp(fs.logreg(Xs, ys, c_k), Xt, classes))
                    add("image only", zi)
                    alpha = fu.support_alpha(Xs, ys, tw, classes, c_k)
                    add(fu.V2, fu.fuse(zt, zi, k, conf=True) if alpha is None else (1 - alpha) * zt + alpha * zi,
                        alpha=alpha)
                    add("Tip-Adapter (default)", tip_logits(Xt, Xs, L, tw, TIP_ALPHA, TIP_BETA))
                    a, b, tuned = tip_support_params(Xs, ys, tw, classes)
                    add("Tip-Adapter (support)", tip_logits(Xt, Xs, L, tw, a, b), tip_alpha=a, tip_beta=b)
            print(f"  {key} / {dname}: done ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "tip_runs.csv", index=False)

    cells = df.groupby(["dataset", "model", "K", "method"]).accuracy.mean().unstack()
    best1 = cells[["text only", "image only"]].max(axis=1)
    out = []
    for m in ["text only", "image only", "Tip-Adapter (default)", "Tip-Adapter (support)", fu.V2]:
        loss = best1 - cells[m]
        r = {"method": m, "n_cells": len(cells), "mean_accuracy": cells[m].mean(), "mean_loss": loss.mean(),
             "worst_loss": loss.max()}
        if m != fu.V2:
            diff = cells[fu.V2] - cells[m]
            r.update({"v2_minus_method": diff.mean(), "v2_wins": int((diff > 0).sum()),
                      "v2_losses": int((diff < 0).sum()),
                      "wilcoxon_p": st.wilcoxon(cells[fu.V2], cells[m]).pvalue if (diff != 0).any() else 1.0})
        out.append(r)
    summ = pd.DataFrame(out)
    summ.round(4).to_csv(OUT / "tip_vs_v2.csv", index=False)
    per_ds = cells.groupby(level="dataset").mean().round(1)
    per_ds.to_csv(OUT / "tip_by_dataset.csv")
    print("\nTip-Adapter vs reliability-weighted fusion (v2), mean over", len(cells), "settings")
    for _, r in summ.iterrows():
        extra = "" if r.method == fu.V2 else \
            f" | v2 {r.v2_minus_method:+.1f} ({int(r.v2_wins)}/{int(r.v2_losses)}) p={r.wilcoxon_p:.2g}"
        print(f"  {r.method:<24} acc {r.mean_accuracy:5.1f} | mean loss {r.mean_loss:5.1f} | "
              f"worst loss {r.worst_loss:5.1f}{extra}")
    print("\nMean accuracy per dataset:\n" + per_ds.to_string())


# ------------------------------------------------------------------ Part B: lab + field
def part_b():
    rows = []
    for key in fm.MODELS:
        for dname in ["plantdoc", "pp"]:
            d = load(key, dname)
            if d is None:
                continue
            (Xt, yt, _), (Xp, yp, pp), classes, all28 = d
            cfg = DATASETS[dname]
            index = {p: i for i, p in enumerate(pp)}
            Xl, yl, _ = lab_data(key, classes)[0]
            c_lab = S.lab_c(key, *fs.load_feats(key, "plantvillage_train_sub")[:2])
            t0 = time.time()

            def acc(clf):
                return 100 * np.mean(clf.predict(Xt) == yt)

            lab_only = acc(LogisticRegression(C=c_lab, max_iter=3000).fit(Xl, yl))
            for k in fs.SHOTS:
                c_k = fs.choose_c_fewshot(*lab_data(key, classes), classes, k)[0]
                for s in fs.SEEDS:
                    Xs, ys = support(cfg, k, s, index, Xp, yp)
                    X2, y2 = np.vstack([Xl, Xs]), np.concatenate([yl, ys])
                    w = np.concatenate([np.ones(len(yl)), np.full(len(ys), len(yl) / len(ys))])
                    base = {"dataset": dname, "model": key, "K": k, "seed": s}
                    rows += [
                        {**base, "method": "lab only", "accuracy": lab_only},
                        {**base, "method": "field only", "accuracy": acc(fs.logreg(Xs, ys, c_k))},
                        {**base, "method": "lab + field",
                         "accuracy": acc(LogisticRegression(C=c_lab, max_iter=3000).fit(X2, y2))},
                        {**base, "method": "lab + field (balanced)",
                         "accuracy": acc(LogisticRegression(C=c_lab, max_iter=3000).fit(X2, y2, sample_weight=w))},
                    ]
            print(f"  {key} / {dname}: done ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "labfield_runs.csv", index=False)
    summ = (df.groupby(["dataset", "model", "K", "method"]).accuracy.agg(["mean", "std"]).round(2)
            .reset_index())
    summ.to_csv(OUT / "labfield_summary.csv", index=False)
    piv = summ.pivot_table(index=["dataset", "model", "K"], columns="method", values="mean")
    piv["best_lab_field_minus_field_only"] = piv[["lab + field", "lab + field (balanced)"]].max(axis=1) - piv["field only"]
    print("\nLab + field vs field only (accuracy %, mean over 5 support sets)")
    print(piv.round(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["A", "B"], choices=["A", "B"])
    args = ap.parse_args()
    if "A" in args.parts:
        print("Part A - Tip-Adapter vs v2")
        part_a()
    if "B" in args.parts:
        print("\nPart B - lab + field training")
        part_b()
    print("\nSaved in results/revision/baselines/")


if __name__ == "__main__":
    main()
