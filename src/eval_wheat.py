"""
Step 10c - New-crop test on JIC wheat (5 classes, never used while designing the method).
Running time: about 5-10 minutes (uses saved features).

Run from the project folder (plant-disease-fm):
    python src/eval_wheat.py

Wheat is absent from PlantVillage, so there is no lab-to-field part; this measures how well each
model recognises the diseases of a NEW crop from text and from K = 1, 5, 10, 16 field photos.

Methods (identical code to Step 9):
  zero-shot text (both prompt styles), prototype, image only (linear probe),
  equal fusion, confidence (CW), CW + SW (v1, kappa = best on PlantDoc + Plant Pathology),
  reliability (v2, ours: declared final method before this dataset was used)

Output: results/wheat/wheat_zeroshot.csv, wheat_runs.csv, wheat_summary.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import fewshot as fs
import fm
import fusion as fu

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

OUT = fm.ROOT / "results" / "wheat"


def wheat_text(key, classes, style):
    cache = fm.FEATURES / key / f"text_wheat_{style}.npy"
    if cache.exists():
        w = np.load(cache)
        if w.shape[0] == len(classes):
            return w
    import zeroshot
    w = zeroshot.text_classifier(key, classes, style)
    np.save(cache, w)
    return w


def v1_kappa(model):
    p = fm.ROOT / "results" / "fusion" / "fusion_runs_all_kappas.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    g = df[(df.model == model) & (df.method == fu.V1)]
    return None if g.empty else float(g.groupby("kappa")["accuracy"].mean().idxmax())


def main():
    models = [k for k in fm.MODELS if (fm.FEATURES / k / "wheat_test.npz").exists()]
    if not models:
        print("No wheat features. Run:  python src/extract_features.py --splits wheat_test wheat_pool --workers 2")
        return
    print("Models:", ", ".join(models))
    zrows, rows = [], []

    for key in models:
        Xt, yt, _ = fs.load_feats(key, "wheat_test")
        Xp, yp, pp = fs.load_feats(key, "wheat_pool")
        classes = sorted(set(yt))
        cls = np.array(classes)
        index = {p: i for i, p in enumerate(pp)}
        name = fm.MODELS[key]["label"]

        zt = None
        if fm.has_text_encoder(key):
            for style in ["common", "scientific"]:
                w = wheat_text(key, classes, style)
                m = fs.metrics(yt, cls[(Xt @ w.T).argmax(1)])
                pre = style == fm.MODELS[key]["prompt"]
                zrows.append({"model": key, "model_name": name, "prompt": style,
                              "pre_specified": int(pre), **m})
                if pre:
                    tw, zt = w, fu.z(Xt @ w.T)
            fm.free_model(key)
            m0 = fs.metrics(yt, cls[zt.argmax(1)])
            for s in fs.SEEDS:
                rows.append({"model": key, "method": "text only", "K": 0, "seed": s, **m0})

        lab_tr = fs.load_feats(key, "plantvillage_train_sub")
        lab_te = fs.load_feats(key, "plantvillage_test_sub")
        all28, _ = fm.class_list()
        kappa = v1_kappa(key) if zt is not None else None

        for k in fs.SHOTS:
            c_k = fs.choose_c_fewshot(lab_tr, lab_te, all28, k)[0]   # C tuned on LAB tasks only
            for s in fs.SEEDS:
                sup = fs.read_csv(fm.SPLITS / f"wheat_fewshot_k{k}_seed{s}.csv")
                idx = [index[r["path"]] for r in sup if r["path"] in index]
                Xs, ys = Xp[idx], yp[idx]

                def add(method, score, alpha=None):
                    rows.append({"model": key, "method": method, "K": k, "seed": s, "alpha": alpha,
                                 **fs.metrics(yt, cls[score.argmax(1)])})

                add("prototype", Xt @ fs.prototypes(Xs, ys, classes).T)
                zi = fu.z(fu.class_logp(fs.logreg(Xs, ys, c_k), Xt, classes))
                add("image only", zi)
                if zt is None:
                    continue
                add("text only", zt)
                add("equal fusion", fu.fuse(zt, zi, k))
                add("confidence (CW)", fu.fuse(zt, zi, k, conf=True))
                if kappa is not None:
                    add(fu.V1, fu.fuse(zt, zi, k, kappa=kappa, conf=True))
                alpha = fu.support_alpha(Xs, ys, tw, classes, c_k)
                if alpha is None:
                    add(fu.V2, fu.fuse(zt, zi, k, conf=True))
                else:
                    add(fu.V2, (1 - alpha) * zt + alpha * zi, alpha)
        print(f"  {key}: done")

    OUT.mkdir(parents=True, exist_ok=True)
    zdf, df = pd.DataFrame(zrows), pd.DataFrame(rows)
    zdf.to_csv(OUT / "wheat_zeroshot.csv", index=False)
    df.to_csv(OUT / "wheat_runs.csv", index=False)
    summ = (df.groupby(["model", "method", "K"])
            .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                 macro_f1_mean=("macro_f1", "mean")).reset_index())
    summ.to_csv(OUT / "wheat_summary.csv", index=False)

    n_test = len(fs.load_feats(models[0], "wheat_test")[1])
    print(f"\nJIC wheat - {n_test} test images, 5 classes (new crop, untouched during method design)")
    if not zdf.empty:
        print("\nZero-shot accuracy %  (* = pre-specified prompt style)")
        for _, r in zdf.iterrows():
            print(f"  {r.model_name:<18} {r.prompt:<11}{'*' if r.pre_specified else ' '} {r.accuracy:6.1f}")
    order = ["text only", "prototype", "image only", "equal fusion", "confidence (CW)", fu.V1, fu.V2]
    for key in models:
        print(f"\n  {fm.MODELS[key]['label']}")
        print(f"    {'method':<23}" + "".join(f"{'K=' + str(k):>8}" for k in [0] + fs.SHOTS) + f"{'mean K>=1':>11}")
        for meth in order:
            g = summ[(summ.model == key) & (summ.method == meth)].set_index("K")
            if g.empty:
                continue
            cells = "".join(f"{g.accuracy_mean[k]:>8.1f}" if k in g.index else f"{'-':>8}" for k in [0] + fs.SHOTS)
            avg = g.loc[[k for k in fs.SHOTS if k in g.index], "accuracy_mean"].mean()
            print(f"    {meth:<23}{cells}{avg:>11.1f}")
    print("\nSaved: results/wheat/")


if __name__ == "__main__":
    main()
