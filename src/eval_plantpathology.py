"""
Step 8c - Evaluate all models on the non-web field test set (Plant Pathology 2020, apple).
Running time: about 5-10 minutes (uses saved features).

Run from the project folder (plant-disease-fm):
    python src/eval_plantpathology.py

Two evaluation settings
  28-way : the model chooses among ALL 28 benchmark classes (realistic: it is not told the
           photo is an apple leaf). Metrics over the 3 apple classes present.
   3-way : the model only chooses among the 3 apple classes (isolates disease recognition).

Experiments
  A. Zero-shot text, both prompt styles (common / scientific)          28-way and 3-way
  B. Lab-to-field: classifier trained on PlantVillage (lab subset)      28-way and 3-way
  C. Few-shot 3-way with K = 1, 5, 10, 16 orchard images per class (5 seeds):
     prototype, linear probe (C tuned on LAB apple tasks only), text+prototype

Output: results/plantpathology/pp_summary.csv, pp_fewshot_runs.csv, pp_fewshot_summary.csv
"""
import csv
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

import fewshot as fs
import fm

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

OUT = fm.ROOT / "results" / "plantpathology"
APPLE = ["Apple___Apple_scab", "Apple___Cedar_apple_rust", "Apple___healthy"]


def text_w(key, classes, style):
    cache = fm.FEATURES / key / f"text_{style}.npy"
    if cache.exists():
        w = np.load(cache)
        if w.shape[0] == len(classes):
            return w
    import zeroshot
    w = zeroshot.text_classifier(key, classes, style)
    np.save(cache, w)
    return w


def restrict(scores, classes, keep):
    idx = [classes.index(c) for c in keep]
    return np.array(keep)[scores[:, idx].argmax(1)]


def main():
    classes, _ = fm.class_list()
    cls = np.array(classes)
    models = [k for k in fm.MODELS if (fm.FEATURES / k / "pp_test.npz").exists()]
    if not models:
        print("No Plant Pathology features found. Run:  python src/extract_features.py --splits pp_test pp_pool")
        return
    print("Models:", ", ".join(models))

    summary, runs = [], []
    for key in models:
        Xt, yt, _ = fs.load_feats(key, "pp_test")
        Xp, yp, pp = fs.load_feats(key, "pp_pool")
        index = {p: i for i, p in enumerate(pp)}
        name = fm.MODELS[key]["label"]

        def add(setting, method, m):
            summary.append({"model": key, "model_name": name, "setting": setting, "method": method, **m})

        # A. zero-shot, both prompt styles
        tw_main = None
        if fm.has_text_encoder(key):
            for style in ["common", "scientific"]:
                w = text_w(key, classes, style)
                s = Xt @ w.T
                tag = "zero-shot " + style + (" (pre-specified)" if style == fm.MODELS[key]["prompt"] else "")
                add("28-way", tag, fs.metrics(yt, cls[s.argmax(1)]))
                add("3-way", tag, fs.metrics(yt, restrict(s, classes, APPLE)))
                if style == fm.MODELS[key]["prompt"]:
                    tw_main = w
            fm.free_model(key)

        # B. lab-to-field
        lab_tr, lab_te = fs.load_feats(key, "plantvillage_train_sub"), fs.load_feats(key, "plantvillage_test_sub")
        if lab_tr is not None:
            c_lab, _ = fs.choose_c_cv(lab_tr[0], lab_tr[1])
            clf = fs.logreg(lab_tr[0], lab_tr[1], c_lab)
            add("28-way", "lab-trained classifier", fs.metrics(yt, clf.predict(Xt)))
            proba = clf.predict_proba(Xt)
            add("3-way", "lab-trained classifier",
                fs.metrics(yt, restrict(proba, list(clf.classes_), APPLE)))

        # C. few-shot, 3-way
        apple_lab_tr = apple_lab_te = None
        if lab_tr is not None:
            m_tr, m_te = np.isin(lab_tr[1], APPLE), np.isin(lab_te[1], APPLE)
            apple_lab_tr = (lab_tr[0][m_tr], lab_tr[1][m_tr], None)
            apple_lab_te = (lab_te[0][m_te], lab_te[1][m_te], None)
        tw3 = tw_main[[classes.index(c) for c in APPLE]] if tw_main is not None else None
        apple = np.array(APPLE)
        if tw3 is not None:
            m0 = fs.metrics(yt, apple[(Xt @ tw3.T).argmax(1)])
            for s in fs.SEEDS:
                runs.append({"model": key, "method": "zero-shot text", "K": 0, "seed": s, **m0})
        for k in fs.SHOTS:
            c_k = fs.choose_c_fewshot(apple_lab_tr, apple_lab_te, APPLE, k)[0] if apple_lab_tr else 1.0
            for s in fs.SEEDS:
                rows = fs.read_csv(fm.SPLITS / f"pp_fewshot_k{k}_seed{s}.csv")
                idx = [index[r["path"]] for r in rows if r["path"] in index]
                Xs, ys = Xp[idx], yp[idx]
                P = fs.prototypes(Xs, ys, APPLE)
                s_p = Xt @ P.T
                runs.append({"model": key, "method": "prototype", "K": k, "seed": s,
                             **fs.metrics(yt, apple[s_p.argmax(1)])})
                runs.append({"model": key, "method": "linear probe", "K": k, "seed": s,
                             **fs.metrics(yt, fs.logreg(Xs, ys, c_k).predict(Xt))})
                if tw3 is not None:
                    s_b = fs.standardise(Xt @ tw3.T) + fs.standardise(s_p)
                    runs.append({"model": key, "method": "text+prototype", "K": k, "seed": s,
                                 **fs.metrics(yt, apple[s_b.argmax(1)])})
        print(f"  {key}: done")

    fs.write(OUT / "pp_summary.csv", summary)
    fs.write(OUT / "pp_fewshot_runs.csv", runs)
    fsum = fs.summarise(runs)
    fs.write(OUT / "pp_fewshot_summary.csv", fsum)

    # ---------- print
    print(f"\nPlant Pathology 2020 (orchard, non-web) - {len(yt)} test images, 3 apple classes")
    for setting in ["28-way", "3-way"]:
        print(f"\n[{setting}]  accuracy % (balanced accuracy)")
        meths = list(dict.fromkeys(r["method"] for r in summary if r["setting"] == setting))
        for meth in meths:
            print(f"  {meth}")
            for key in models:
                r = [x for x in summary if x["model"] == key and x["setting"] == setting and x["method"] == meth]
                if r:
                    print(f"    {fm.MODELS[key]['label']:<20} {r[0]['accuracy']:5.1f}  ({r[0]['balanced_accuracy']:4.1f})")
    print("\n[Few-shot, 3-way]  mean ± std over 5 repeats")
    for meth in ["zero-shot text", "prototype", "linear probe", "text+prototype"]:
        rows = [r for r in fsum if r["method"] == meth]
        if not rows:
            continue
        ks = sorted({r["K"] for r in rows})
        print(f"\n  {meth}")
        print(f"  {'Model':<20}" + "".join(f"{'K=' + str(k):>13}" for k in ks))
        for m in dict.fromkeys(r["model"] for r in rows):
            cells = {r["K"]: r for r in rows if r["model"] == m}
            print(f"  {fm.MODELS[m]['label']:<20}" + "".join(
                f"{cells[k]['accuracy_mean']:>7.1f} ±{cells[k]['accuracy_std']:>4.1f}" if k in cells else f"{'-':>13}"
                for k in ks))
    print("\nSaved: results/plantpathology/")


if __name__ == "__main__":
    main()
