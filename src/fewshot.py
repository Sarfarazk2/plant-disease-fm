"""
Step 7 - Lab-to-field gap and few-shot experiments (uses the features saved in Step 6).
Running time: about 5-15 minutes for all models (no images are re-processed).

Run from the project folder (plant-disease-fm):
    python src/fewshot.py                      # all models that have features ready
    python src/fewshot.py --models clip_b16    # one model

Experiments (all on the PlantDoc field test set, 27 classes present, 28-way prediction):
  A. Lab-to-field gap  : linear classifier trained on PlantVillage (lab) features only,
                         tested on PlantVillage test (lab) AND PlantDoc test (field).
  B. Few-shot (K = 1, 5, 10, 16 field images per class, 5 random repeats each):
       prototype      - nearest class mean (no training)
       linear_probe   - logistic regression on the K field images
       text+prototype - zero-shot text scores + prototype scores (CLIP-style models only)
     K = 0 for CLIP-style models = zero-shot with text prompts, using each model's
     pre-specified prompt style (fm.MODELS[...]["prompt"]).

Output:
    results/fewshot/lab_to_field.csv        lab accuracy, field accuracy, drop
    results/fewshot/fewshot_runs.csv        every single run (model, method, K, seed)
    results/fewshot/fewshot_summary.csv     mean, std and 95% CI over the 5 repeats
    figures/fig_fewshot_curves.(svg/pdf/eps/png/tiff) + _data.csv
"""
import argparse
import csv
import warnings

import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import fm

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
# expected at K=1 (one image per class), harmless:
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

OUT = fm.ROOT / "results" / "fewshot"
SHOTS = [1, 5, 10, 16]
SEEDS = [0, 1, 2, 3, 4]
# Regularisation strength C is chosen WITHOUT field data:
#   lab classifier      -> 3-fold cross-validation on lab training features
#   few-shot classifier -> simulated K-shot tasks drawn from LAB data (seeds 100-104)
C_GRID = [0.1, 1, 10, 100, 1000, 10000]
TUNE_SEEDS = [100, 101, 102, 103, 104]
N_WAY = 5           # literature-style N-way K-shot episodes (secondary evaluation)
N_EPISODES = 600


# ------------------------------------------------------------------ helpers
def load_feats(key, split):
    p = fm.FEATURES / key / f"{split}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return d["feats"], np.array(list(d["labels"])), list(d["paths"])


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metrics(y_true, y_pred):
    present = sorted(set(y_true))
    return {
        "accuracy": 100 * accuracy_score(y_true, y_pred),
        "balanced_accuracy": 100 * balanced_accuracy_score(y_true, y_pred),
        "macro_f1": 100 * f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
    }


def text_weights(key, classes):
    """Zero-shot text classifier using the model's pre-specified prompt style; cached on disk."""
    style = fm.MODELS[key]["prompt"]
    cache = fm.FEATURES / key / f"text_{style}.npy"
    if cache.exists():
        w = np.load(cache)
        if w.shape[0] == len(classes):
            return w
    import zeroshot
    w = zeroshot.text_classifier(key, classes, style)
    np.save(cache, w)
    fm.free_model(key)
    return w


def standardise(scores):
    """Per-image z-score, so text and image similarities (different scales) can be added
    with equal weight and no tuned mixing hyperparameter."""
    return (scores - scores.mean(1, keepdims=True)) / (scores.std(1, keepdims=True) + 1e-8)


def prototypes(Xs, ys, classes):
    P = np.stack([Xs[ys == c].mean(axis=0) for c in classes])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def logreg(X, y, C):
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(X, y)
    return clf


def choose_c_cv(X, y):
    """C for the lab classifier: 3-fold stratified cross-validation on lab training data."""
    import time
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    print(f"    choosing C by 3-fold cross-validation on {len(y)} lab images "
          f"({len(C_GRID)} values; large sets can take several minutes) ...", flush=True)
    scores = {}
    for c in C_GRID:
        t0 = time.time()
        scores[c] = cross_val_score(LogisticRegression(C=c, max_iter=3000), X, y, cv=cv,
                                    n_jobs=-1).mean()
        print(f"      C={c:<6} cv accuracy {100 * scores[c]:.1f}%  ({time.time() - t0:.0f} s)",
              flush=True)
    return max(scores, key=scores.get), scores


def sample_k(y, classes, k, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in classes:
        pool = np.flatnonzero(y == c)
        if len(pool):
            idx += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.array(idx)


def choose_c_fewshot(lab_tr, lab_te, classes, k):
    """C for the K-shot linear probe, tuned on simulated K-shot tasks from LAB data only."""
    Xl, yl, _ = lab_tr
    scores = {}
    for c in C_GRID:
        accs = []
        for s in TUNE_SEEDS:
            idx = sample_k(yl, classes, k, s)
            accs.append(accuracy_score(lab_te[1], logreg(Xl[idx], yl[idx], c).predict(lab_te[0])))
        scores[c] = float(np.mean(accs))
    return max(scores, key=scores.get), scores


def episodic(Xp, yp, Xt, yt, classes, k, tw=None, seed=0):
    """Literature-style N-way K-shot episodes: support from the field train pool, queries from
    the field test set, prototype classifier (and text+prototype when text weights exist)."""
    rng = np.random.default_rng(seed)
    usable = [c for c in classes if (yp == c).sum() >= k and (yt == c).sum() >= 1]
    cidx = {c: i for i, c in enumerate(classes)}
    acc_p, acc_tp = [], []
    for _ in range(N_EPISODES):
        way = list(rng.choice(usable, size=N_WAY, replace=False))
        sup = np.concatenate([rng.choice(np.flatnonzero(yp == c), size=k, replace=False) for c in way])
        qry = np.concatenate([np.flatnonzero(yt == c) for c in way])
        P = prototypes(Xp[sup], yp[sup], way)
        s_p = Xt[qry] @ P.T
        truth = yt[qry]
        acc_p.append(np.mean(np.array(way)[s_p.argmax(1)] == truth))
        if tw is not None:
            s_t = Xt[qry] @ tw[[cidx[c] for c in way]].T
            acc_tp.append(np.mean(np.array(way)[(standardise(s_t) + standardise(s_p)).argmax(1)] == truth))
    out = {"prototype": acc_p}
    if acc_tp:
        out["text+prototype"] = acc_tp
    return out


# ------------------------------------------------------------------ experiments
def run_model(key, classes, runs, lab_rows, lab="sub", c_rows=None, epi_rows=None):
    test = load_feats(key, "plantdoc_test")
    pool = load_feats(key, "plantdoc_train_clean")
    if test is None or pool is None:
        print(f"  {key}: field features missing - run Step 6 first, skipping")
        return
    Xt, yt, _ = test
    Xp, yp, pp = pool
    index = {p: i for i, p in enumerate(pp)}
    cls_arr = np.array(classes)

    # ---- A. lab-to-field gap (balanced lab subset by default; "full" = all lab images)
    sfx = "_sub" if lab == "sub" else ""
    lab_tr = load_feats(key, f"plantvillage_train{sfx}")
    lab_te = load_feats(key, f"plantvillage_test{sfx}")
    if lab_tr is not None and lab_te is not None:
        c_lab, _ = choose_c_cv(lab_tr[0], lab_tr[1])
        clf = logreg(lab_tr[0], lab_tr[1], c_lab)
        lab_m = metrics(lab_te[1], clf.predict(lab_te[0]))
        field = metrics(yt, clf.predict(Xt))
        lab_rows.append({"model": key, "model_name": fm.MODELS[key]["label"], "lab_data": lab,
                         "C": c_lab, "n_lab_train": len(lab_tr[1]), "n_lab_test": len(lab_te[1]),
                         "lab_accuracy": lab_m["accuracy"], "field_accuracy": field["accuracy"],
                         "field_macro_f1": field["macro_f1"],
                         "drop_points": lab_m["accuracy"] - field["accuracy"]})
        print(f"  {key}: lab {lab_m['accuracy']:.1f}%  ->  field {field['accuracy']:.1f}%  "
              f"(lab data: {lab}, {len(lab_tr[1])} train images, C={c_lab})")
    else:
        print(f"  {key}: PlantVillage ({lab}) features not ready yet - lab-to-field part skipped")
    if lab == "full":
        return
    if lab_tr is None or lab_te is None:
        print(f"  {key}: few-shot linear probe needs the lab subset features to choose C - skipped")
        return

    # ---- B. few-shot
    tw = text_weights(key, classes) if fm.has_text_encoder(key) else None
    if tw is not None:
        m = metrics(yt, cls_arr[(Xt @ tw.T).argmax(1)])
        for s in SEEDS:   # deterministic, repeated so every K has 5 rows
            runs.append({"model": key, "method": "zero-shot text", "K": 0, "seed": s, **m})

    for k in SHOTS:
        c_k, tune = choose_c_fewshot(lab_tr, lab_te, classes, k)
        for s in SEEDS:
            rows = read_csv(fm.SPLITS / f"fewshot_k{k}_seed{s}.csv")
            idx = [index[r["path"]] for r in rows if r["path"] in index]
            Xs, ys = Xp[idx], yp[idx]

            P = prototypes(Xs, ys, classes)
            s_proto = Xt @ P.T
            runs.append({"model": key, "method": "prototype", "K": k, "seed": s,
                         **metrics(yt, cls_arr[s_proto.argmax(1)])})

            # supplementary transparency: field accuracy for EVERY C (not used for selection)
            for c in C_GRID:
                acc = 100 * accuracy_score(yt, logreg(Xs, ys, c).predict(Xt))
                if c_rows is not None:
                    c_rows.append({"model": key, "K": k, "seed": s, "C": c,
                                   "lab_tuning_accuracy": 100 * tune[c], "field_accuracy": acc,
                                   "selected": int(c == c_k)})
            clf = logreg(Xs, ys, c_k)
            runs.append({"model": key, "method": "linear probe", "K": k, "seed": s,
                         **metrics(yt, clf.predict(Xt))})

            if tw is not None:
                s_both = standardise(Xt @ tw.T) + standardise(s_proto)
                runs.append({"model": key, "method": "text+prototype", "K": k, "seed": s,
                             **metrics(yt, cls_arr[s_both.argmax(1)])})
        print(f"  {key}: K={k:>2} done (linear-probe C={c_k}, chosen on lab data)")

    # ---- C. literature-style 5-way episodes (secondary, for comparison with prior work)
    if epi_rows is not None:
        for k in [1, 5]:
            res = episodic(Xp, yp, Xt, yt, classes, k, tw)
            for meth, accs in res.items():
                a = 100 * np.array(accs)
                epi_rows.append({"model": key, "model_name": fm.MODELS[key]["label"],
                                 "method": meth, "N_way": N_WAY, "K": k, "episodes": len(a),
                                 "accuracy_mean": a.mean(),
                                 "ci95": 1.96 * a.std(ddof=1) / np.sqrt(len(a))})
    print(f"  {key}: few-shot done")


def summarise(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["model"], r["method"], r["K"]), []).append(r)
    out = []
    for (m, meth, k), rs in sorted(groups.items(), key=lambda g: (g[0][0], g[0][1], g[0][2])):
        row = {"model": m, "model_name": fm.MODELS[m]["label"], "method": meth, "K": k, "n_runs": len(rs)}
        for met in ["accuracy", "balanced_accuracy", "macro_f1"]:
            v = np.array([r[met] for r in rs])
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            half = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
            row[f"{met}_mean"], row[f"{met}_std"] = v.mean(), sd
            row[f"{met}_ci95_low"], row[f"{met}_ci95_high"] = v.mean() - half, v.mean() + half
        out.append(row)
    return out


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})


def plot(summary):
    import matplotlib.pyplot as plt
    import pandas as pd
    from figstyle import apply_style, save_figure

    from figstyle import COLORS

    apply_style()
    df = pd.DataFrame(summary)
    # one fixed colour and marker per model, identical in every panel and every figure
    style = {m: (COLORS[i % len(COLORS)], "osD^v"[i % 5]) for i, m in enumerate(fm.MODELS)}
    methods = ["prototype", "linear probe", "text+prototype"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    handles = {}
    for ax, meth in zip(axes, methods):
        for m in [k for k in fm.MODELS if k in set(df["model"])]:
            d = df[(df.model == m) & (df.method == meth)].sort_values("K")
            z = df[(df.model == m) & (df.method == "zero-shot text")]
            if meth == "text+prototype" and not z.empty:
                d = pd.concat([z, d])
            if d.empty:
                continue
            color, marker = style[m]
            h = ax.errorbar(d["K"], d["accuracy_mean"], yerr=d["accuracy_std"], color=color,
                            marker=marker, capsize=2, label=fm.MODELS[m]["label"])
            handles[m] = h
        ax.set_title(meth)
        ax.set_xlabel("Field images per class (K)")
        ax.set_xticks([0, 1, 5, 10, 16])
    axes[0].set_ylabel("Accuracy on PlantDoc test (%)")
    fig.legend([handles[m] for m in handles], [fm.MODELS[m]["label"] for m in handles],
               loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, "fig_fewshot_curves",
                data=df[["model_name", "method", "K", "accuracy_mean", "accuracy_std",
                         "macro_f1_mean", "macro_f1_std"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, choices=list(fm.MODELS),
                    help="default: every model whose field features already exist")
    ap.add_argument("--lab", default="sub", choices=["sub", "full"],
                    help="lab data for the lab-to-field gap: balanced subset (default) or all images")
    args = ap.parse_args()
    if args.models is None:
        args.models = [k for k in fm.MODELS
                       if (fm.FEATURES / k / "plantdoc_test.npz").exists()
                       and (fm.FEATURES / k / "plantdoc_train_clean.npz").exists()]
        print("Models with features ready:", ", ".join(args.models) or "none")

    classes, _ = fm.class_list()
    runs, lab_rows, c_rows, epi_rows = [], [], [], []

    if args.lab == "full":   # robustness check only: do not overwrite the main few-shot results
        for key in args.models:
            run_model(key, classes, runs, lab_rows, "full")
        if lab_rows:
            write(OUT / "lab_to_field_full.csv", lab_rows)
            for r in lab_rows:
                print(f"{r['model_name']:<18} FULL lab data: lab {r['lab_accuracy']:5.1f}%  "
                      f"field {r['field_accuracy']:5.1f}%  drop {r['drop_points']:5.1f} points")
            print("Saved: results/fewshot/lab_to_field_full.csv")
        return

    for key in args.models:
        run_model(key, classes, runs, lab_rows, "sub", c_rows, epi_rows)
    if not runs:
        print("No features found. Run Step 6 (extract_features.py) first.")
        return

    write(OUT / "fewshot_runs.csv", runs)
    summary = summarise(runs)
    write(OUT / "fewshot_summary.csv", summary)
    if lab_rows:
        write(OUT / "lab_to_field.csv", lab_rows)
    if c_rows:
        write(OUT / "linear_probe_C_sensitivity.csv", c_rows)
    if epi_rows:
        write(OUT / "episodic_5way.csv", epi_rows)
    plot(summary)

    print("\nFew-shot accuracy on PlantDoc test (mean +/- std over 5 repeats)")
    for meth in ["zero-shot text", "prototype", "linear probe", "text+prototype"]:
        rows = [r for r in summary if r["method"] == meth]
        if not rows:
            continue
        print(f"\n[{meth}]")
        ks = sorted({r["K"] for r in rows})
        print(f"{'Model':<18}" + "".join(f"{'K=' + str(k):>14}" for k in ks))
        for m in dict.fromkeys(r["model"] for r in rows):
            cells = {r["K"]: r for r in rows if r["model"] == m}
            print(f"{fm.MODELS[m]['label']:<18}" + "".join(
                f"{cells[k]['accuracy_mean']:>8.1f} ±{cells[k]['accuracy_std']:>4.1f}" if k in cells
                else f"{'-':>14}" for k in ks))
    if lab_rows:
        print("\nLab-to-field gap (classifier trained on PlantVillage only)")
        for r in lab_rows:
            print(f"{r['model_name']:<18} lab {r['lab_accuracy']:5.1f}%  field {r['field_accuracy']:5.1f}%"
                  f"  drop {r['drop_points']:5.1f} points  (C={r['C']})")
    if epi_rows:
        print(f"\n{N_WAY}-way episodes on PlantDoc (for comparison with prior work; mean ± 95% CI)")
        for r in epi_rows:
            print(f"{r['model_name']:<18} {r['method']:<15} {N_WAY}-way {r['K']}-shot: "
                  f"{r['accuracy_mean']:5.1f} ± {r['ci95']:.1f}")
    print("\nSaved: results/fewshot/ and figures/fig_fewshot_curves.*")


if __name__ == "__main__":
    main()
