"""
Step 16 - PlantDoc label audit: how much do the main PlantDoc results change after fixing labels?
Running time: about 5-10 minutes (CPU, saved features).

Run from the project folder (plant-disease-fm), after placing plantdoc_label_audit.csv in
results/analysis/:
    python src/rescore_plantdoc.py

The audit file (results/analysis/plantdoc_label_audit.csv) lists PlantDoc images whose label is
  confirmed error         - the ORIGINAL file name (from the web source) names another disease or
                            crop, e.g. 'corn-gray-leaf-spot-f4.jpg' filed under northern leaf blight
  probable error (visual) - flagged by >= 4 of 5 models AND judged wrong on visual inspection
  questionable            - mixed / unclear photos (e.g. healthy + iron-deficient leaf together)
For test images a corrected label is given when the true class is one of the 28 classes;
otherwise (e.g. southern rust, bacterial canker, iron deficiency) the image is excluded.

Test-set versions
  original   : 236 images, labels as published
  corrected  : confirmed errors fixed (or removed when the true class is not in the 28)
  strict     : corrected + probable and questionable images removed
Few-shot also checked with CLEAN SUPPORTS: confirmed-error training images removed from the
K = 16 support sets (so a class may get slightly fewer than 16 images).

Output: results/analysis/rescore_plantdoc.csv, rescore_plantdoc_summary.txt
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import fewshot as fs
import fm
import fusion as fu

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

OUT = fm.ROOT / "results" / "analysis"
K = 16


def key(path):
    """Match paths on <split>/<folder>/<file> (robust to different root folders)."""
    return "/".join(str(path).replace("\\", "/").split("/")[-3:])


def versions(paths, y, audit, short2label):
    a = audit[audit.split == "test"].set_index("k")
    out = {"original": (np.ones(len(y), bool), y.copy())}
    yc, keep_c, keep_s = y.copy(), np.ones(len(y), bool), np.ones(len(y), bool)
    for i, p in enumerate(paths):
        k = key(p)
        if k not in a.index:
            continue
        r = a.loc[k]
        if r.category == "confirmed error":
            if isinstance(r.corrected_label, str) and r.corrected_label.strip():
                yc[i] = short2label[r.corrected_label.strip()]
            else:
                keep_c[i] = False
            keep_s[i] = keep_c[i]
        else:                                   # probable / questionable -> only removed in "strict"
            keep_s[i] = False
    out["corrected"] = (keep_c, yc)
    out["strict"] = (keep_s, yc)
    return out


def main():
    all28, short = fm.class_list()
    short2label = {v: k for k, v in short.items()}
    cls = np.array(all28)
    audit = pd.read_csv(OUT / "plantdoc_label_audit.csv")
    audit["k"] = audit.path.map(key)
    bad_train = set(audit[(audit.split == "train") & (audit.category == "confirmed error")].k)
    lf = pd.read_csv(fm.ROOT / "results" / "fewshot" / "lab_to_field.csv")

    rows = []
    models = [m for m in fm.MODELS if (fm.FEATURES / m / "plantdoc_test.npz").exists()]
    for m in models:
        Xt, yt, pt = fs.load_feats(m, "plantdoc_test")
        Xp, yp, pp = fs.load_feats(m, "plantdoc_train_clean")
        lab_tr, lab_te = fs.load_feats(m, "plantvillage_train_sub"), fs.load_feats(m, "plantvillage_test_sub")
        V = versions(pt, yt, audit, short2label)
        preds = {}
        g = lf[(lf.model == m) & (lf.lab_data == "sub")]
        c_lab = float(g.C.iloc[0]) if not g.empty else fs.choose_c_cv(lab_tr[0], lab_tr[1])[0]
        preds["lab-trained (frozen + linear)"] = [fs.logreg(lab_tr[0], lab_tr[1], c_lab).predict(Xt)]
        text = fm.has_text_encoder(m)
        tw = fs.text_weights(m, all28) if text else None
        if text:
            preds["zero-shot text"] = [cls[(Xt @ tw.T).argmax(1)]]
        c_k = fs.choose_c_fewshot(lab_tr, lab_te, all28, K)[0]
        index = {key(p): i for i, p in enumerate(pp)}
        for clean in [False, True]:
            tag = " clean supports" if clean else ""
            lp, v2 = [], []
            for s in fs.SEEDS:
                sup = fs.read_csv(fm.SPLITS / f"fewshot_k{K}_seed{s}.csv")
                idx = [index[key(r["path"])] for r in sup if key(r["path"]) in index
                       and not (clean and key(r["path"]) in bad_train)]
                Xs, ys = Xp[idx], yp[idx]
                clf = fs.logreg(Xs, ys, c_k)
                zi = fu.z(fu.class_logp(clf, Xt, all28))
                lp.append(cls[zi.argmax(1)])
                if text:
                    zt = fu.z(Xt @ tw.T)
                    a = fu.support_alpha(Xs, ys, tw, all28, c_k)
                    f = fu.fuse(zt, zi, K, conf=True) if a is None else (1 - a) * zt + a * zi
                    v2.append(cls[f.argmax(1)])
            preds[f"K={K} linear probe{tag}"] = lp
            if text:
                preds[f"K={K} v2 fusion{tag}"] = v2
        for meth, plist in preds.items():
            for vname, (keep, y) in V.items():
                accs = [100 * np.mean(p[keep] == y[keep]) for p in plist]
                rows.append({"model": fm.MODELS[m]["label"], "method": meth, "test_version": vname,
                             "n_test": int(keep.sum()), "accuracy": float(np.mean(accs))})
        fm.free_model(m)
        print(f"  {m}: done")

    # fully fine-tuned CNNs (per-image outputs saved by stats.py)
    for m in fm.CNNS:
        p = fm.ROOT / "results" / "stats" / f"finetuned_{m}_logits.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        pt, y = list(z["paths_plantdoc_test"]), cls[z["y_plantdoc_test"]]
        pred = cls[z["logits_plantdoc_test"].argmax(1)]
        for vname, (keep, yv) in versions(pt, y, audit, short2label).items():
            rows.append({"model": fm.MODELS[m]["label"] + " (fine-tuned)", "method": "lab-trained (fully fine-tuned)",
                         "test_version": vname, "n_test": int(keep.sum()),
                         "accuracy": 100 * float(np.mean(pred[keep] == yv[keep]))})

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(OUT / "rescore_plantdoc.csv", index=False)
    piv = df.pivot_table(index=["method", "model"], columns="test_version", values="accuracy", sort=False)
    piv = piv[["original", "corrected", "strict"]]
    piv["change (strict - original)"] = piv["strict"] - piv["original"]
    n = df.groupby("test_version").n_test.max()
    lines = [f"PlantDoc test accuracy (%) before and after the label audit "
             f"(n = {n['original']} original, {n['corrected']} corrected, {n['strict']} strict)", ""]
    for meth, g in piv.groupby(level=0, sort=False):
        lines.append(f"[{meth}]")
        g = g.droplevel(0).sort_values("original", ascending=False)
        for mdl, r in g.iterrows():
            lines.append(f"  {mdl:<34} {r.original:5.1f} -> {r.corrected:5.1f} -> {r.strict:5.1f}   "
                         f"({r['change (strict - original)']:+.1f})")
        rank_o = list(g.sort_values("original", ascending=False).index)
        rank_s = list(g.sort_values("strict", ascending=False).index)
        lines.append(f"  ranking of models unchanged: {'yes' if rank_o == rank_s else 'NO'}")
        lines.append("")
    tr = audit[audit.split == "train"]
    lines.append(f"Training pool: {int((tr.category == 'confirmed error').sum())} confirmed label errors "
                 f"and {int((tr.category == 'questionable').sum())} questionable images found from file names "
                 f"(lower bound; most web file names carry no disease information).")
    text = "\n".join(lines)
    (OUT / "rescore_plantdoc_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print("Saved: results/analysis/rescore_plantdoc.csv, rescore_plantdoc_summary.txt")


if __name__ == "__main__":
    main()
