"""
Step 9 - Adaptive text + image fusion (the paper's method contribution).
Running time: about 5-15 minutes (uses saved features only).

Run from the project folder (plant-disease-fm):
    python src/fusion.py

Problem: equal-weight fusion of zero-shot text scores and few-shot image scores can be WORSE
than text alone (e.g. BioCLIP 2 on orchard photos: 79.0% text-only -> 71.6% with 1 image).

Methods compared (all use the same text classifier and the same few-shot linear probe):
  text only          zero-shot text scores
  image only         few-shot linear probe (C tuned on lab data only, as in Step 7)
  equal fusion       z(text) + z(image)
  confidence (CW)    per image, each source weighted by 1/entropy of its own prediction
                     -> no hyperparameter
  sample-size (SW)   (1-a_K) z(text) + a_K z(image),  a_K = K / (K + kappa)
  CW + SW (v1)       per-image confidence weights multiplied by the sample-size prior
                     (kappa chosen CROSS-DATASET: tuned on PlantDoc, applied to Plant
                     Pathology and vice versa). Did not transfer reliably (see v2).
  reliability (v2)   FINAL METHOD, declared before running: the image weight alpha is
                     estimated from the K labelled support images only - text accuracy on
                     them (direct) vs image accuracy on them (leave-one-out); alpha = grid
                     value with best support accuracy. No hyperparameter is carried across
                     datasets and no test image is used. K = 1 falls back to CW.

Output: results/fusion/fusion_runs.csv, fusion_summary.csv, kappa_selection.csv
        figures/fig_fusion.(svg/pdf/eps/png/tiff) + _data.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import fewshot as fs
import fm

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

OUT = fm.ROOT / "results" / "fusion"
KAPPAS = [0.5, 1, 2, 4, 8, 16, 32, 64]
APPLE = ["Apple___Apple_scab", "Apple___Cedar_apple_rust", "Apple___healthy"]
DATASETS = {
    "plantdoc": {"label": "PlantDoc (web)", "test": "plantdoc_test", "pool": "plantdoc_train_clean",
                 "fewshot": "fewshot_k{k}_seed{s}", "classes": None},
    "pp": {"label": "Plant Pathology (orchard)", "test": "pp_test", "pool": "pp_pool",
           "fewshot": "pp_fewshot_k{k}_seed{s}", "classes": APPLE},
}


def z(s):
    return fs.standardise(s)


def softmax(s):
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def entropy_weight(zs):
    p = softmax(zs)
    h = -(p * np.log(p + 1e-12)).sum(1, keepdims=True)
    return 1.0 / (h + 1e-6)


def fuse(zt, zi, k, kappa=None, conf=False):
    a = 0.5 if kappa is None else k / (k + kappa)
    wt, wi = (1 - a), a
    if conf:
        ct, ci = entropy_weight(zt), entropy_weight(zi)
        wt, wi = wt * ct, wi * ci
        tot = wt + wi
        wt, wi = wt / tot, wi / tot
    return wt * zt + wi * zi


def run_dataset(dname, cfg, model, rows):
    all_classes, _ = fm.class_list()
    classes = cfg["classes"] or all_classes
    cls = np.array(classes)
    test, pool = fs.load_feats(model, cfg["test"]), fs.load_feats(model, cfg["pool"])
    if test is None or pool is None:
        return False
    Xt, yt, _ = test
    Xp, yp, pp = pool
    index = {p: i for i, p in enumerate(pp)}

    tw = fs.text_weights(model, all_classes)[[all_classes.index(c) for c in classes]]
    zt = z(Xt @ tw.T)
    m0 = fs.metrics(yt, cls[zt.argmax(1)])
    for s in fs.SEEDS:
        rows.append({"dataset": dname, "model": model, "method": "text only", "kappa": None,
                     "K": 0, "seed": s, **m0})

    lab_tr = fs.load_feats(model, "plantvillage_train_sub")
    lab_te = fs.load_feats(model, "plantvillage_test_sub")
    if cfg["classes"]:
        mtr, mte = np.isin(lab_tr[1], classes), np.isin(lab_te[1], classes)
        lab_tr, lab_te = (lab_tr[0][mtr], lab_tr[1][mtr], None), (lab_te[0][mte], lab_te[1][mte], None)

    for k in fs.SHOTS:
        c_k = fs.choose_c_fewshot(lab_tr, lab_te, classes, k)[0]
        for s in fs.SEEDS:
            sup = fs.read_csv(fm.SPLITS / (cfg["fewshot"].format(k=k, s=s) + ".csv"))
            idx = [index[r["path"]] for r in sup if r["path"] in index]
            Xs, ys = Xp[idx], yp[idx]
            clf = fs.logreg(Xs, ys, c_k)
            zi = z(class_logp(clf, Xt, classes))

            def add(method, score, kappa=None, alpha=None):
                rows.append({"dataset": dname, "model": model, "method": method, "kappa": kappa,
                             "alpha": alpha, "K": k, "seed": s,
                             **fs.metrics(yt, cls[score.argmax(1)])})

            add("text only", zt)
            add("image only", zi)
            add("equal fusion", fuse(zt, zi, k))
            add("confidence (CW)", fuse(zt, zi, k, conf=True))
            for kap in KAPPAS:
                add("sample-size (SW)", fuse(zt, zi, k, kappa=kap), kap)
                add(V1, fuse(zt, zi, k, kappa=kap, conf=True), kap)

            # ---- v2: reliability measured on the K support images themselves
            alpha = support_alpha(Xs, ys, tw, classes, c_k)
            if alpha is None:                      # K = 1: no hold-out possible -> CW fallback
                add(V2, fuse(zt, zi, k, conf=True))
            else:
                add(V2, (1 - alpha) * zt + alpha * zi, alpha=alpha)
    return True


V1 = "CW + SW (v1)"
V2 = "reliability (v2, ours)"
ALPHAS = np.round(np.linspace(0, 1, 11), 2)


def class_logp(clf, X, classes):
    """Log-probabilities in the fixed class order (classes absent from training get -30)."""
    logp = np.full((len(X), len(classes)), -30.0)
    proba = clf.predict_log_proba(X)
    for j, c in enumerate(clf.classes_):
        logp[:, classes.index(c)] = proba[:, j]
    return logp


def support_alpha(Xs, ys, tw, classes, c):
    """v2 weight of the image branch, estimated ONLY from the K labelled support images:
    text scores on the support images are computed directly (text never saw them); image
    scores are computed leave-one-out (fold f holds out the f-th image of every class).
    alpha = the grid value giving the best support accuracy (ties -> closest to 0.5).
    Returns None when K = 1 (hold-out impossible)."""
    by_class = {c_: np.flatnonzero(ys == c_) for c_ in classes}
    k_max = max(len(v) for v in by_class.values())
    if k_max < 2:
        return None
    zi_sup = np.zeros((len(ys), len(classes)))
    for f in range(k_max):
        held = np.array([v[f] for v in by_class.values() if len(v) > f])
        keep = np.setdiff1d(np.arange(len(ys)), held)
        if len(np.unique(ys[keep])) < 2:
            return None
        clf = fs.logreg(Xs[keep], ys[keep], c)
        zi_sup[held] = z(class_logp(clf, Xs[held], classes))
    zt_sup = z(Xs @ tw.T)
    truth = np.array([classes.index(y) for y in ys])
    accs = np.array([np.mean(((1 - a) * zt_sup + a * zi_sup).argmax(1) == truth) for a in ALPHAS])
    best = np.flatnonzero(accs == accs.max())
    return float(ALPHAS[best[np.argmin(np.abs(ALPHAS[best] - 0.5))]])


def select_kappa(df):
    """kappa for dataset D = the value with best mean accuracy (over K and seeds) on the OTHER dataset."""
    sel, out = [], []
    for model in df.model.unique():
        for meth in ["sample-size (SW)", V1]:
            for d in DATASETS:
                other = [o for o in DATASETS if o != d][0]
                g = df[(df.model == model) & (df.method == meth) & (df.dataset == other)]
                if g.empty:
                    continue
                best = g.groupby("kappa")["accuracy"].mean().idxmax()
                sel.append({"model": model, "method": meth, "applied_to": d, "tuned_on": other,
                            "kappa": best})
                out.append(df[(df.model == model) & (df.method == meth) & (df.dataset == d)
                              & (df.kappa == best)])
    fixed = df[~df.method.isin(["sample-size (SW)", V1])]
    return pd.concat([fixed] + out, ignore_index=True), pd.DataFrame(sel)


def plot(summ):
    import matplotlib.pyplot as plt
    from figstyle import COLORS, apply_style, save_figure

    apply_style()
    methods = ["text only", "image only", "equal fusion", V2]
    style = {"text only": (COLORS[1], "--", "s"), "image only": (COLORS[2], "-", "^"),
             "equal fusion": (COLORS[7], ":", "o"), V2: (COLORS[0], "-", "D")}
    models = [m for m in fm.MODELS if m in set(summ.model)]
    fig, axes = plt.subplots(len(DATASETS), len(models), figsize=(7.2, 3.6), sharex=True,
                             squeeze=False)
    for r, d in enumerate(DATASETS):
        for c, m in enumerate(models):
            ax = axes[r, c]
            for meth in methods:
                g = summ[(summ.dataset == d) & (summ.model == m) & (summ.method == meth)].sort_values("K")
                if g.empty:
                    continue
                col, ls, mk = style[meth]
                ax.errorbar(g.K, g.accuracy_mean, yerr=g.accuracy_std, color=col, ls=ls, marker=mk,
                            capsize=2, label=meth, markersize=3)
            if r == 0:
                ax.set_title(fm.MODELS[m]["label"], fontsize=7)
            if c == 0:
                ax.set_ylabel(f"{DATASETS[d]['label']}\naccuracy (%)", fontsize=7)
            if r == len(DATASETS) - 1:
                ax.set_xlabel("K", fontsize=7)
            ax.set_xticks([0, 1, 5, 10, 16])
            ax.tick_params(labelsize=6)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, "fig_fusion", data=summ)


def main():
    models = [k for k in fm.MODELS if fm.has_text_encoder(k)
              and (fm.FEATURES / k / "plantdoc_test.npz").exists()]
    print("Models:", ", ".join(models))
    rows = []
    for m in models:
        for d, cfg in DATASETS.items():
            if run_dataset(d, cfg, m, rows):
                print(f"  {m:<11} | {d}: done")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "fusion_runs_all_kappas.csv", index=False)

    final, sel = select_kappa(df)
    sel.to_csv(OUT / "kappa_selection.csv", index=False)
    final.to_csv(OUT / "fusion_runs.csv", index=False)
    summ = (final.groupby(["dataset", "model", "method", "K"])
            .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                 macro_f1_mean=("macro_f1", "mean"), n=("accuracy", "size")).reset_index())
    summ.to_csv(OUT / "fusion_summary.csv", index=False)
    plot(summ)

    order = ["text only", "image only", "equal fusion", "confidence (CW)", "sample-size (SW)", V1, V2]
    for d in DATASETS:
        print(f"\n=== {DATASETS[d]['label']} ===  accuracy % (mean over 5 repeats)")
        for m in models:
            print(f"\n  {fm.MODELS[m]['label']}")
            print(f"    {'method':<23}" + "".join(f"{'K=' + str(k):>8}" for k in [0] + fs.SHOTS)
                  + f"{'mean K>=1':>11}")
            for meth in order:
                g = summ[(summ.dataset == d) & (summ.model == m) & (summ.method == meth)].set_index("K")
                if g.empty:
                    continue
                cells = "".join(f"{g.accuracy_mean[k]:>8.1f}" if k in g.index else f"{'-':>8}"
                                for k in [0] + fs.SHOTS)
                avg = g.loc[[k for k in fs.SHOTS if k in g.index], "accuracy_mean"].mean()
                print(f"    {meth:<23}{cells}{avg:>11.1f}")
    print("\nkappa chosen on the other dataset (v1):")
    print(sel.to_string(index=False))
    a = final[(final.method == V2) & final.alpha.notna()]
    if not a.empty:
        print("\nv2: mean image weight alpha chosen from the support images (0 = text only, 1 = image only)")
        print(a.pivot_table(index=["dataset", "model"], columns="K", values="alpha", aggfunc="mean")
              .round(2).to_string())
    print("\nSaved: results/fusion/ and figures/fig_fusion.*")


if __name__ == "__main__":
    main()
