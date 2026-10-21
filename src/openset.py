"""
Step 11 - Unknown-disease (open-set) detection.
Running time: about 5-10 minutes (uses saved features).

Run from the project folder (plant-disease-fm):
    python src/openset.py

Question: can a model tell that a field photo shows a disease it does NOT know?

Two settings (K = 16 field photos per KNOWN class, 5 repeats):
  near-unknown : 7 of the 28 PlantDoc classes are hidden (a different random split in every
                 repeat). Known = PlantDoc test images of the 21 known classes,
                 unknown = PlantDoc test images of the 7 hidden classes.
  far-unknown  : all 28 classes known; unknown = JIC wheat test images (a crop never seen).

"Known-ness" scores (higher = more likely a known class):
  max text similarity   - largest cosine to any known class text (zero-shot, no photos)
  prototype             - largest cosine to any known class prototype
  linear probe (MSP)    - largest predicted probability of the few-shot linear probe
  v2 fusion (MSP)       - largest softmax of the v2 fused scores (Step 9 method)
Metrics: AUROC (50 = random, 100 = perfect) and FPR@95 (fraction of unknown photos accepted
as known when 95% of known photos are accepted; lower is better). Closed-set accuracy on the
known classes is reported alongside.

Output: results/openset/openset_runs.csv, openset_summary.csv
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

import fewshot as fs
import fm
import fusion as fu

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

OUT = fm.ROOT / "results" / "openset"
K = 16
N_HIDDEN = 7


def fpr_at_95(known_scores, unknown_scores):
    thr = np.percentile(known_scores, 5)          # 95% of known photos score above thr
    return 100 * np.mean(unknown_scores >= thr)


def scores_for(key, Xs, ys, known, Xq, tw_all, all_classes, c_k):
    """Return {method: (known-ness score per query, predicted class per query)}."""
    known = list(known)
    kn = np.array(known)
    out = {}
    P = fs.prototypes(Xs, ys, known)
    s_p = Xq @ P.T
    out["prototype"] = (s_p.max(1), kn[s_p.argmax(1)])
    clf = fs.logreg(Xs, ys, c_k)
    logp = fu.class_logp(clf, Xq, known)
    out["linear probe (MSP)"] = (np.exp(logp).max(1), kn[logp.argmax(1)])
    if tw_all is not None:
        tw = tw_all[[all_classes.index(c) for c in known]]
        s_t = Xq @ tw.T
        out["max text similarity"] = (s_t.max(1), kn[s_t.argmax(1)])
        zt, zi = fu.z(s_t), fu.z(logp)
        alpha = fu.support_alpha(Xs, ys, tw, known, c_k)
        fused = fu.fuse(zt, zi, K, conf=True) if alpha is None else (1 - alpha) * zt + alpha * zi
        out["v2 fusion (MSP)"] = (fu.softmax(fused).max(1), kn[fused.argmax(1)])
    return out


def main():
    all_classes, _ = fm.class_list()
    models = [k for k in fm.MODELS if (fm.FEATURES / k / "plantdoc_test.npz").exists()
              and (fm.FEATURES / k / "plantdoc_train_clean.npz").exists()]
    print("Models:", ", ".join(models))
    rows = []
    for key in models:
        Xt, yt, _ = fs.load_feats(key, "plantdoc_test")
        Xp, yp, pp = fs.load_feats(key, "plantdoc_train_clean")
        wheat = fs.load_feats(key, "wheat_test")
        index = {p: i for i, p in enumerate(pp)}
        tw_all = fs.text_weights(key, all_classes) if fm.has_text_encoder(key) else None
        lab_tr = fs.load_feats(key, "plantvillage_train_sub")
        lab_te = fs.load_feats(key, "plantvillage_test_sub")
        c_k = fs.choose_c_fewshot(lab_tr, lab_te, all_classes, K)[0]
        present = sorted(set(yt))                      # classes that have test images

        for s in fs.SEEDS:
            sup = fs.read_csv(fm.SPLITS / f"fewshot_k{K}_seed{s}.csv")
            idx = np.array([index[r["path"]] for r in sup if r["path"] in index])

            # ---- near-unknown: hide 7 classes (only classes that have test images)
            rng = np.random.default_rng(1000 + s)
            hidden = set(rng.choice(present, size=N_HIDDEN, replace=False))
            known = [c for c in all_classes if c not in hidden]
            m_sup = np.isin(yp[idx], known)
            m_kn, m_un = np.isin(yt, known), np.isin(yt, list(hidden))
            Xq = np.concatenate([Xt[m_kn], Xt[m_un]])
            is_known = np.r_[np.ones(m_kn.sum()), np.zeros(m_un.sum())]
            res = scores_for(key, Xp[idx][m_sup], yp[idx][m_sup], known, Xq, tw_all, all_classes, c_k)
            for meth, (sc, pred) in res.items():
                rows.append({"setting": "near-unknown", "model": key, "method": meth, "seed": s,
                             "auroc": 100 * roc_auc_score(is_known, sc),
                             "fpr95": fpr_at_95(sc[is_known == 1], sc[is_known == 0]),
                             "known_acc": 100 * np.mean(pred[: m_kn.sum()] == yt[m_kn])})

            # ---- far-unknown: wheat photos vs all 28 known classes
            if wheat is not None:
                Xq = np.concatenate([Xt, wheat[0]])
                is_known = np.r_[np.ones(len(yt)), np.zeros(len(wheat[1]))]
                res = scores_for(key, Xp[idx], yp[idx], all_classes, Xq, tw_all, all_classes, c_k)
                for meth, (sc, pred) in res.items():
                    rows.append({"setting": "far-unknown (wheat)", "model": key, "method": meth, "seed": s,
                                 "auroc": 100 * roc_auc_score(is_known, sc),
                                 "fpr95": fpr_at_95(sc[is_known == 1], sc[is_known == 0]),
                                 "known_acc": 100 * np.mean(pred[: len(yt)] == yt)})
        fm.free_model(key)
        print(f"  {key}: done")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "openset_runs.csv", index=False)
    summ = (df.groupby(["setting", "model", "method"])
            .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
                 fpr95_mean=("fpr95", "mean"), known_acc_mean=("known_acc", "mean")).reset_index())
    summ.to_csv(OUT / "openset_summary.csv", index=False)

    order = ["max text similarity", "prototype", "linear probe (MSP)", "v2 fusion (MSP)"]
    for setting in summ.setting.unique():
        print(f"\n=== {setting} ===   AUROC % (higher better) | FPR@95 % (lower better) | known-class acc %")
        for key in models:
            print(f"\n  {fm.MODELS[key]['label']}")
            for meth in order:
                g = summ[(summ.setting == setting) & (summ.model == key) & (summ.method == meth)]
                if g.empty:
                    continue
                r = g.iloc[0]
                print(f"    {meth:<22} AUROC {r.auroc_mean:5.1f} ±{r.auroc_std:4.1f} | "
                      f"FPR95 {r.fpr95_mean:5.1f} | acc {r.known_acc_mean:5.1f}")
    print("\nSaved: results/openset/")


if __name__ == "__main__":
    main()
