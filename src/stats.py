"""
Step 14 - Statistics: 95% confidence intervals and significance tests for the main claims.
Running time: about 15-25 minutes (CPU, saved features). If the fine-tuned CNN from Step 13
exists, about 1-2 extra minutes on the GPU to get its per-image predictions.
Run AFTER finetune_cnn.py has finished - not at the same time.

Run from the project folder (plant-disease-fm):
    python src/stats.py

Question answered: "is model/method A really better than B, or could it be luck?"

How
  * Every prediction is recomputed per test image from the saved features, with exactly the
    same code and the same choices (C, alpha) as Steps 7-10. No field data chooses anything.
  * Few-shot: each test image gets its correctness averaged over the 5 support repeats.
  * 95% CI of accuracy: bootstrap over test images (2000 resamples, percentile method).
  * A vs B on the same test images: paired sign-flip permutation test on the per-image
    differences (10,000 permutations, two-sided) + paired-bootstrap 95% CI of the difference.
    Holm correction for multiple comparisons inside each table.
  * Fusion v2 over ALL cases: Wilcoxon signed-rank test over every (dataset, model, K) cell,
    v2 vs each baseline, plus mean and worst loss against the better of text-only/image-only.

Output (results/stats/):
    accuracy_ci.csv          every result with its 95% CI (and the std over the 5 repeats)
    vs_best.csv              in every table: best model vs each other model (diff, CI, p, Holm p)
    fusion_v2_tests.csv      v2 vs text only / image only / equal fusion in every cell
    fusion_v2_overall.csv    Wilcoxon test over all cells + mean/worst loss
    controlled_pairs.csv     BioCLIP vs CLIP ViT-B/16 (OpenAI) and BioCLIP 2 vs CLIP ViT-L/14 (LAION-2B):
                             same architecture and starting weights, only the biology training differs
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.exceptions import ConvergenceWarning

import fewshot as fs
import fm
import fusion as fu

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

OUT = fm.ROOT / "results" / "stats"
N_BOOT, N_PERM = 2000, 10000
APPLE = fu.APPLE
DATA = {
    "PlantDoc": {"test": "plantdoc_test", "pool": "plantdoc_train_clean", "fewshot": "fewshot_k{k}_seed{s}"},
    "Plant Pathology": {"test": "pp_test", "pool": "pp_pool", "fewshot": "pp_fewshot_k{k}_seed{s}"},
    "JIC wheat": {"test": "wheat_test", "pool": "wheat_pool", "fewshot": "wheat_fewshot_k{k}_seed{s}"},
}
FUSION = ["text only", "image only", "equal fusion", fu.V2]
OTHER = {"common": "scientific", "scientific": "common"}
# Controlled pairs: biology model vs the exact general checkpoint it was trained from
PAIRS = [("bioclip", "clip_b16"), ("bioclip2", "clip_l14")]


# ------------------------------------------------------------------ statistics helpers
_BOOT = {}


def boot_idx(n):
    """The SAME resamples are used for every model on a test set -> comparisons are paired."""
    if n not in _BOOT:
        _BOOT[n] = np.random.default_rng(n).integers(0, n, size=(N_BOOT, n), dtype=np.int32)
    return _BOOT[n]


def ci(c):
    m = c[boot_idx(len(c))].mean(1)
    return 100 * c.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def paired(ca, cb):
    d = ca - cb
    obs = d.mean()
    bm = d[boot_idx(len(d))].mean(1)
    rng = np.random.default_rng(12345)
    hits, done = 0, 0
    while done < N_PERM:
        n = min(1000, N_PERM - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(d)))
        hits += int(np.sum(np.abs((signs * d).mean(1)) >= abs(obs) - 1e-12))
        done += n
    p = (hits + 1) / (N_PERM + 1)
    return 100 * obs, 100 * np.percentile(bm, 2.5), 100 * np.percentile(bm, 97.5), p


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ------------------------------------------------------------------ data helpers
ORDER = {}


def feats(key, split):
    """Features re-ordered to one fixed image order per split (so models are paired)."""
    d = fs.load_feats(key, split)
    if d is None:
        return None
    X, y, paths = d
    if split not in ORDER:
        ORDER[split] = sorted(paths)
    pos = {p: i for i, p in enumerate(paths)}
    idx = np.array([pos[p] for p in ORDER[split]])
    return X[idx], y[idx]


def lab_c(key, X, y):
    p = fm.ROOT / "results" / "fewshot" / "lab_to_field.csv"
    if p.exists():
        df = pd.read_csv(p)
        g = df[(df.model == key) & (df.lab_data == "sub")]
        if not g.empty:
            return float(g.C.iloc[0])
    return fs.choose_c_cv(X, y)[0]


REC = []


def add(table, dataset, task, k, model, method, correct, seed_accs=None):
    REC.append({"table": table, "dataset": dataset, "task": task, "K": k, "model": model,
                "method": method, "correct": np.asarray(correct, float),
                "seed_std": float(np.std(seed_accs, ddof=1)) if seed_accs is not None else 0.0})


# ------------------------------------------------------------------ recompute predictions
def run_model(key, all28):
    import eval_plantpathology
    import eval_wheat
    cls28 = np.array(all28)
    text = fm.has_text_encoder(key)
    tw28 = fs.text_weights(key, all28) if text else None
    lab_tr = fs.load_feats(key, "plantvillage_train_sub")
    lab_te = fs.load_feats(key, "plantvillage_test_sub")

    # ---- zero-shot (pre-specified prompt) and lab-trained classifier, 28-way
    clf = fs.logreg(lab_tr[0], lab_tr[1], lab_c(key, lab_tr[0], lab_tr[1]))
    for d in ["PlantDoc", "Plant Pathology"]:
        Xt, yt = feats(key, DATA[d]["test"])
        if text:
            add("zero-shot", d, "28-way", 0, key, "zero-shot text", cls28[(Xt @ tw28.T).argmax(1)] == yt)
            w_alt = eval_plantpathology.text_w(key, all28, OTHER[fm.MODELS[key]["prompt"]])
            add("zero-shot (other prompt)", d, "28-way", 0, key, "zero-shot text (other prompt)",
                cls28[(Xt @ w_alt.T).argmax(1)] == yt)
        add("lab-trained", d, "28-way", None, key, "frozen + linear", clf.predict(Xt) == yt)

    # ---- few-shot on the three field datasets
    for d, cfg in DATA.items():
        test, pool = feats(key, cfg["test"]), fs.load_feats(key, cfg["pool"])
        if test is None or pool is None:
            continue
        Xt, yt = test
        Xp, yp, pp = pool
        index = {p: i for i, p in enumerate(pp)}
        if d == "PlantDoc":
            classes, task, ltr, lte = all28, "28-way", lab_tr, lab_te
            tw = tw28
        elif d == "Plant Pathology":
            classes, task = APPLE, "3-way"
            mtr, mte = np.isin(lab_tr[1], APPLE), np.isin(lab_te[1], APPLE)
            ltr, lte = (lab_tr[0][mtr], lab_tr[1][mtr], None), (lab_te[0][mte], lab_te[1][mte], None)
            tw = tw28[[all28.index(c) for c in APPLE]] if text else None
        else:
            classes, task, ltr, lte = sorted(set(yt)), "5-way", lab_tr, lab_te
            tw = eval_wheat.wheat_text(key, classes, fm.MODELS[key]["prompt"]) if text else None
        cls = np.array(classes)
        zt = fu.z(Xt @ tw.T) if text else None
        if text and d == "JIC wheat":
            add("zero-shot", d, task, 0, key, "zero-shot text", cls[zt.argmax(1)] == yt)
            w_alt = eval_wheat.wheat_text(key, classes, OTHER[fm.MODELS[key]["prompt"]])
            add("zero-shot (other prompt)", d, task, 0, key, "zero-shot text (other prompt)",
                cls[(Xt @ w_alt.T).argmax(1)] == yt)
        # C: same rule as Steps 7-10 (simulated K-shot tasks on LAB data; all 28 classes for wheat)
        c_classes = APPLE if d == "Plant Pathology" else all28

        for k in fs.SHOTS:
            c_k = fs.choose_c_fewshot(ltr, lte, c_classes, k)[0]
            corr = {}
            for s in fs.SEEDS:
                sup = fs.read_csv(fm.SPLITS / (cfg["fewshot"].format(k=k, s=s) + ".csv"))
                idx = [index[r["path"]] for r in sup if r["path"] in index]
                Xs, ys = Xp[idx], yp[idx]
                pred = {"prototype": cls[(Xt @ fs.prototypes(Xs, ys, classes).T).argmax(1)]}
                zi = fu.z(fu.class_logp(fs.logreg(Xs, ys, c_k), Xt, classes))
                pred["image only"] = cls[zi.argmax(1)]
                if text:
                    pred["text only"] = cls[zt.argmax(1)]
                    pred["equal fusion"] = cls[fu.fuse(zt, zi, k).argmax(1)]
                    a = fu.support_alpha(Xs, ys, tw, classes, c_k)
                    f = fu.fuse(zt, zi, k, conf=True) if a is None else (1 - a) * zt + a * zi
                    pred[fu.V2] = cls[f.argmax(1)]
                for m, p in pred.items():
                    corr.setdefault(m, []).append(p == yt)
            for m, cs in corr.items():
                cs = np.array(cs, float)
                add("few-shot", d, task, k, key, m, cs.mean(0), 100 * cs.mean(1))
    fm.free_model(key)


def run_finetuned(all28):
    """Per-image predictions of the fully fine-tuned CNN(s) from Step 13 (if finished)."""
    done = [k for k in fm.CNNS if (fm.ROOT / "models" / f"{k}_lab_best.pt").exists()
            and (fm.ROOT / "results" / "finetune" / f"finetune_{k}.csv").exists()]
    if not done:
        print("  (no finished fine-tuned CNN found - skipped; run finetune_cnn.py first to include it)")
        return
    import timm
    import torch
    from torchvision import transforms as T

    import finetune_cnn as ft
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(ft.MEAN, ft.STD)])
    for key in done:
        model = timm.create_model(fm.MODELS[key]["name"], pretrained=False, num_classes=len(all28))
        model.load_state_dict(torch.load(fm.ROOT / "models" / f"{key}_lab_best.pt",
                                         map_location="cpu", weights_only=True))
        model = model.to(fm.DEVICE).eval()
        saved = {}
        for d in ["PlantDoc", "Plant Pathology", "lab"]:
            split = DATA[d]["test"] if d in DATA else "plantvillage_test_sub"
            by_path = {r["path"]: r for r in fm.read_split(split)}
            order = ORDER.get(split) or sorted(by_path)
            rows = [by_path[p] for p in order]
            logits, y = ft.predict(model, rows, all28, tf, 32, 0)
            saved.update({f"logits_{split}": logits, f"y_{split}": y,
                          f"paths_{split}": np.array(order, dtype=object)})
            if d in DATA:
                add("lab-trained", d, "28-way", None, key, "fully fine-tuned", logits.argmax(1) == y)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / f"finetuned_{key}_logits.npz", **saved)   # used by analysis.py (calibration)
        del model
        if fm.DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"  {key} (fully fine-tuned): done")


# ------------------------------------------------------------------ tables
def name(key):
    return fm.MODELS[key]["label"]


def build_tables():
    rec = pd.DataFrame(REC)
    rec["Kc"] = rec.K.fillna(-1)

    # 1. every result with its CI
    rows = []
    for r in REC:
        acc, lo, hi = ci(r["correct"])
        rows.append({**{k: r[k] for k in ["table", "dataset", "task", "K", "method"]},
                     "model": name(r["model"]), "n_test": len(r["correct"]), "accuracy": acc,
                     "ci95_low": lo, "ci95_high": hi, "std_over_repeats": r["seed_std"]})
    acc_ci = pd.DataFrame(rows)

    # 2. best model vs every other model (same table, dataset, K, comparable method)
    groups = [("zero-shot", None), ("lab-trained", None), ("few-shot", "image only"), ("few-shot", fu.V2)]
    vs = []
    for table, meth in groups:
        sub = rec[(rec.table == table) & ((rec.method == meth) if meth else True)]
        for (d, kc), g in sub.groupby(["dataset", "Kc"], sort=False):
            g = g.assign(acc=g.correct.apply(np.mean)).sort_values("acc", ascending=False)
            best = g.iloc[0]
            part = []
            for _, o in g.iloc[1:].iterrows():
                diff, lo, hi, p = paired(best.correct, o.correct)
                part.append({"table": table, "dataset": d, "task": best.task,
                             "K": None if kc < 0 else int(kc), "best": f"{name(best.model)} ({best.method})",
                             "other": f"{name(o.model)} ({o.method})", "best_acc": 100 * best.acc,
                             "other_acc": 100 * o.acc, "diff": diff, "diff_ci95_low": lo,
                             "diff_ci95_high": hi, "p": p})
            if part:
                for x, ph in zip(part, holm([x["p"] for x in part])):
                    x["p_holm"], x["sig"] = ph, stars(ph)
                vs += part
    vs = pd.DataFrame(vs)

    # 3. fusion v2 vs baselines in every (dataset, model, K) cell
    fz = rec[(rec.table == "few-shot") & rec.method.isin(FUSION)]
    cells = []
    for (d, m, k), g in fz.groupby(["dataset", "model", "K"], sort=False):
        c = {r.method: r.correct for r in g.itertuples()}
        if fu.V2 not in c:
            continue
        for base in ["text only", "image only", "equal fusion"]:
            diff, lo, hi, p = paired(c[fu.V2], c[base])
            cells.append({"dataset": d, "model": name(m), "K": int(k), "baseline": base,
                          "v2_acc": 100 * c[fu.V2].mean(), "baseline_acc": 100 * c[base].mean(),
                          "diff": diff, "diff_ci95_low": lo, "diff_ci95_high": hi, "p": p})
    cells = pd.DataFrame(cells)
    for base in cells.baseline.unique():
        m = cells.baseline == base
        cells.loc[m, "p_holm"] = holm(cells.loc[m, "p"])
    cells["sig"] = cells.p_holm.apply(stars)

    # 4. overall: Wilcoxon over cells + loss vs the better single source
    piv = cells.pivot_table(index=["dataset", "model", "K"], columns="baseline", values="baseline_acc")
    piv["v2"] = cells.groupby(["dataset", "model", "K"]).v2_acc.first()
    piv["eq"] = piv["equal fusion"]
    oracle = piv[["text only", "image only"]].max(axis=1)
    overall = []
    for meth, col in [("reliability (v2, ours)", "v2"), ("image only", "image only"),
                      ("text only", "text only"), ("equal fusion", "eq")]:
        loss = oracle - piv[col]
        row = {"method": meth, "n_cells": len(piv), "mean_accuracy": piv[col].mean(),
               "mean_loss_vs_better_single_source": loss.mean(), "worst_loss": loss.max()}
        if col != "v2":
            w = st.wilcoxon(piv["v2"], piv[col])
            row.update({"v2_minus_this_mean": (piv["v2"] - piv[col]).mean(),
                        "v2_wins": int((piv["v2"] > piv[col]).sum()),
                        "v2_losses": int((piv["v2"] < piv[col]).sum()), "wilcoxon_p": w.pvalue})
        overall.append(row)
    # 5. controlled pairs: same architecture + same starting weights, only biology training differs
    pairs = []
    for bio, base in PAIRS:
        a, b = rec[rec.model == bio], rec[rec.model == base]
        if a.empty or b.empty:
            continue
        for r in a.itertuples():
            o = b[(b.table == r.table) & (b.dataset == r.dataset) & (b.Kc == r.Kc) & (b.method == r.method)]
            if o.empty:
                continue
            diff, lo, hi, p = paired(r.correct, o.correct.iloc[0])
            pairs.append({"pair": f"{name(bio)} vs {name(base)}", "table": r.table, "dataset": r.dataset,
                          "task": r.task, "K": None if r.Kc < 0 else int(r.Kc), "method": r.method,
                          "bio_acc": 100 * r.correct.mean(), "general_acc": 100 * o.correct.iloc[0].mean(),
                          "diff": diff, "diff_ci95_low": lo, "diff_ci95_high": hi, "p": p})
    pairs = pd.DataFrame(pairs)
    if not pairs.empty:
        for pr in pairs.pair.unique():
            m = pairs.pair == pr
            pairs.loc[m, "p_holm"] = holm(pairs.loc[m, "p"])
        pairs["sig"] = pairs.p_holm.apply(stars)
    return acc_ci, vs, cells, pd.DataFrame(overall), pairs


def save(df, fname):
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT / fname, index=False)


def main():
    all28, _ = fm.class_list()
    models = [k for k in fm.MODELS if all((fm.FEATURES / k / f"{DATA[d]['test']}.npz").exists() for d in DATA)]
    print("Models:", ", ".join(models))
    for key in models:
        run_model(key, all28)
        print(f"  {key}: done")
    run_finetuned(all28)

    print("\nComputing bootstrap CIs and permutation tests ...")
    acc_ci, vs, cells, overall, pairs = build_tables()
    save(pairs, "controlled_pairs.csv")
    save(acc_ci, "accuracy_ci.csv")
    save(vs, "vs_best.csv")
    save(cells, "fusion_v2_tests.csv")
    save(overall, "fusion_v2_overall.csv")

    # ---------------- short printout of the key claims
    pd.set_option("display.width", 200)
    print("\n=== Lab-trained classifier tested in the field (28-way): best vs others ===")
    for _, r in vs[vs.table == "lab-trained"].iterrows():
        print(f"  {r.dataset:<16} {r.best:<45} {r.best_acc:5.1f} vs {r.other:<45} {r.other_acc:5.1f} | "
              f"diff {r['diff']:+5.1f} [{r.diff_ci95_low:+5.1f}, {r.diff_ci95_high:+5.1f}] {r.sig}")
    print("\n=== Zero-shot: best vs others ===")
    for _, r in vs[vs.table == "zero-shot"].iterrows():
        print(f"  {r.dataset:<16} {r.best:<38} {r.best_acc:5.1f} vs {r.other:<38} {r.other_acc:5.1f} | "
              f"diff {r['diff']:+5.1f} [{r.diff_ci95_low:+5.1f}, {r.diff_ci95_high:+5.1f}] {r.sig}")
    print("\n=== Few-shot linear probe, K = 16: best vs others ===")
    g = vs[(vs.table == "few-shot") & (vs.K == 16) & vs.best.str.contains("image only")]
    for _, r in g.iterrows():
        print(f"  {r.dataset:<16} {r.best:<34} {r.best_acc:5.1f} vs {r.other:<42} {r.other_acc:5.1f} | "
              f"diff {r['diff']:+5.1f} [{r.diff_ci95_low:+5.1f}, {r.diff_ci95_high:+5.1f}] {r.sig}")
    if not pairs.empty:
        print("\n=== Controlled pairs: biology training vs the SAME general model it started from ===")
        for pr, g in pairs.groupby("pair", sort=False):
            print(f"\n  {pr}")
            for _, r in g.iterrows():
                if r.table == "few-shot" and r.method not in ("image only", fu.V2):
                    continue
                k = "" if pd.isna(r.K) else f"K={int(r.K)}"
                print(f"    {r.table:<24} {r.dataset:<16} {k:<5} {r.method:<30} {r.bio_acc:5.1f} vs "
                      f"{r.general_acc:5.1f} | diff {r['diff']:+5.1f} [{r.diff_ci95_low:+5.1f}, "
                      f"{r.diff_ci95_high:+5.1f}] {r.sig}")
    print("\n=== Fusion v2 over all cells (dataset x model x K) ===")
    for _, r in overall.iterrows():
        line = (f"  {r.method:<24} mean acc {r.mean_accuracy:5.1f} | mean loss vs better single source "
                f"{r.mean_loss_vs_better_single_source:4.1f} | worst loss {r.worst_loss:4.1f}")
        if not pd.isna(r.get("wilcoxon_p", np.nan)):
            line += (f" | v2 better by {r.v2_minus_this_mean:+4.1f} (wins {int(r.v2_wins)}, losses {int(r.v2_losses)}),"
                     f" Wilcoxon p = {r.wilcoxon_p:.3g}")
        print(line)
    sig = cells.groupby("baseline").apply(lambda g: pd.Series({
        "v2 significantly better": int(((g.sig != "n.s.") & (g["diff"] > 0)).sum()),
        "v2 significantly worse": int(((g.sig != "n.s.") & (g["diff"] < 0)).sum()),
        "no significant difference": int((g.sig == "n.s.").sum())}), include_groups=False)
    print("\n  Cells where v2 differs significantly from each baseline (Holm-corrected, p < 0.05):")
    print(sig.to_string())
    print("\n  * p<0.05  ** p<0.01  *** p<0.001  n.s. = not significant (after Holm correction)")
    print("Saved: results/stats/")


if __name__ == "__main__":
    main()
