"""
Step 15 - Error analysis, calibration, PlantDoc label audit and the "lab vs field" figure.
Running time: about 5-10 minutes (CPU, saved features). Run after stats.py.

Run from the project folder (plant-disease-fm):
    python src/analysis.py

Parts
  A. fig_lab_vs_field   : slope chart. Every model looks almost the same in the lab, but they
                          spread out widely in the field (frozen models + fully fine-tuned CNNs).
  B. fig_perclass       : per-disease accuracy on PlantDoc (models x classes heatmap):
                          classifier trained on lab images, and K = 16 field images per class.
  C. fig_confusion      : confusion matrix of BioCLIP 2 (lab-trained) on PlantDoc + top confusions.
  D. fig_calibration    : are models confidently wrong in the field? Expected calibration error
                          (ECE) and mean confidence vs accuracy, lab vs field.
  E. Label audit        : PlantDoc test photos where most foundation models agree on a
                          DIFFERENT class than the given label -> candidate label errors, saved
                          with a contact sheet for manual checking (fill in "your_decision").

Output: figures/fig_lab_vs_field.*, fig_perclass.*, fig_confusion.*, fig_calibration.*
        results/analysis/*.csv, results/analysis/label_audit_sheet.png
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

import fewshot as fs
import fm
from figstyle import COLORS, apply_style, save_figure

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

OUT = fm.ROOT / "results" / "analysis"
K_AUDIT = 16
AUDIT_MIN_AGREE = 4          # of the 5 audit models below
# fixed list (decided before the ViT-L controls were added) so the audit list does not change
AUDIT_MODELS = ["clip_b16", "bioclip", "bioclip2", "siglip_b16", "dinov2_b14"]
FT_LABEL = " (fine-tuned)"


def short_model(key):
    return fm.MODELS[key]["label"].replace(" (ImageNet)", "")


def lab_c(key, X, y):
    p = fm.ROOT / "results" / "fewshot" / "lab_to_field.csv"
    if p.exists():
        df = pd.read_csv(p)
        g = df[(df.model == key) & (df.lab_data == "sub")]
        if not g.empty:
            return float(g.C.iloc[0])
    return fs.choose_c_cv(X, y)[0]


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return 100 * e


def spread(y, gap):
    """Move label positions apart (keeps order) so text does not overlap."""
    y = np.asarray(y, float)
    order = np.argsort(y)
    ys = y[order]
    clusters = [[i] for i in range(len(ys))]          # groups of labels that touch each other

    def place(c):
        centre = ys[c].mean()
        return centre - (len(c) - 1) / 2 * gap + gap * np.arange(len(c))

    merged = True
    while merged:
        merged = False
        for j in range(1, len(clusters)):
            if place(clusters[j - 1])[-1] + gap > place(clusters[j])[0]:
                clusters[j - 1] += clusters.pop(j)
                merged = True
                break
    adj = np.concatenate([place(c) for c in clusters])
    out = np.empty_like(adj)
    out[order] = adj
    return out


# ------------------------------------------------------------------ A. lab vs field
def part_a():
    rows = []
    lf = pd.read_csv(fm.ROOT / "results" / "fewshot" / "lab_to_field.csv")
    lf = lf[lf.lab_data == "sub"]
    pps = pd.read_csv(fm.ROOT / "results" / "plantpathology" / "pp_summary.csv")
    pps = pps[(pps.setting == "28-way") & (pps.method == "lab-trained classifier")]
    for _, r in lf.iterrows():
        pp = pps[pps.model == r.model]
        rows.append({"model": short_model(r.model), "kind": "frozen CNN" if fm.MODELS[r.model].get("cnn") else "frozen foundation model",
                     "lab": r.lab_accuracy, "PlantDoc": r.field_accuracy,
                     "Plant Pathology": float(pp.accuracy.iloc[0]) if not pp.empty else np.nan})
    for key in fm.CNNS:
        p = fm.ROOT / "results" / "finetune" / f"finetune_{key}.csv"
        if not p.exists():
            continue
        f = pd.read_csv(p)
        get = lambda data: float(f[(f.data == data) & (f.setting == "28-way")].accuracy.iloc[0])
        rows.append({"model": short_model(key) + FT_LABEL, "kind": "fine-tuned CNN",
                     "lab": get("PlantVillage lab (sub)"), "PlantDoc": get("PlantDoc field"),
                     "Plant Pathology": get("Plant Pathology field")})
    df = pd.DataFrame(rows)

    apply_style()
    style = {"frozen foundation model": (COLORS[0], "o", "-"), "frozen CNN": (COLORS[3], "s", "-"),
             "fine-tuned CNN": (COLORS[3], "D", "--")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, field in zip(axes, ["PlantDoc", "Plant Pathology"]):
        d = df.dropna(subset=[field])
        for _, r in d.iterrows():
            c, mk, ls = style[r.kind]
            ax.plot([0, 1], [r.lab, r[field]], color=c, ls=ls, lw=1, marker=mk, ms=3.5,
                    mfc="white" if r.kind == "fine-tuned CNN" else c)
        ly = spread(d[field].values, 2.8)
        for (_, r), y in zip(d.iterrows(), ly):
            ax.annotate(r.model, xy=(1, r[field]), xytext=(1.06, y), fontsize=6, va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.3, color="0.6"))
        rho = st.spearmanr(d.lab, d[field]).statistic
        ax.set_xticks([0, 1], ["Lab test\n(PlantVillage)", f"Field test\n({field})"])
        ax.set_xlim(-0.15, 1.75)
        ax.set_title(f"{field}: lab range {d.lab.max() - d.lab.min():.1f} pts, "
                     f"field range {d[field].max() - d[field].min():.1f} pts\n"
                     f"(Spearman rank correlation lab vs field = {rho:.2f})", fontsize=7)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Accuracy (%) - classifier trained on lab images only")
    handles = [plt.Line2D([], [], color=c, ls=ls, marker=mk, mfc="white" if k == "fine-tuned CNN" else c, label=k)
               for k, (c, mk, ls) in style.items() if k in set(df.kind)]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, "fig_lab_vs_field", data=df)
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(OUT / "lab_vs_field.csv", index=False)
    print("\nA. Lab vs field (lab-trained classifiers)")
    print(df.round(1).to_string(index=False))
    for field in ["PlantDoc", "Plant Pathology"]:
        d = df.dropna(subset=[field])
        rho, p = st.spearmanr(d.lab, d[field])
        print(f"   {field}: lab spread {d.lab.max() - d.lab.min():.1f} pts, field spread "
              f"{d[field].max() - d[field].min():.1f} pts, Spearman rho = {rho:.2f} (p = {p:.2f}, n = {len(d)})")


# ------------------------------------------------------------------ B-E. per-image analyses
def collect(all28):
    """Per model: lab-trained probabilities (lab test, PlantDoc, PP) and K=16 field predictions."""
    res = {}
    for key in fm.MODELS:
        if not (fm.FEATURES / key / "plantdoc_test.npz").exists():
            continue
        lab_tr, lab_te = fs.load_feats(key, "plantvillage_train_sub"), fs.load_feats(key, "plantvillage_test_sub")
        Xt, yt, paths = fs.load_feats(key, "plantdoc_test")
        Xp, yp, pp = fs.load_feats(key, "plantdoc_train_clean")
        clf = fs.logreg(lab_tr[0], lab_tr[1], lab_c(key, lab_tr[0], lab_tr[1]))
        r = {"classes": list(clf.classes_), "pd_y": yt, "pd_paths": paths,
             "pd_prob": clf.predict_proba(Xt), "lab_prob": clf.predict_proba(lab_te[0]), "lab_y": lab_te[1]}
        ppt = fs.load_feats(key, "pp_test")
        if ppt is not None:
            r["pp_prob"], r["pp_y"] = clf.predict_proba(ppt[0]), ppt[1]
        # K = 16 field linear probe (same C rule as Step 7), predictions for each of the 5 repeats
        c16 = fs.choose_c_fewshot(lab_tr, lab_te, all28, K_AUDIT)[0]
        index = {p: i for i, p in enumerate(pp)}
        preds, probs = [], []
        for s in fs.SEEDS:
            sup = fs.read_csv(fm.SPLITS / f"fewshot_k{K_AUDIT}_seed{s}.csv")
            idx = [index[x["path"]] for x in sup if x["path"] in index]
            m = fs.logreg(Xp[idx], yp[idx], c16)
            preds.append(m.predict(Xt))
            pr = np.zeros((len(Xt), len(all28)))
            pr[:, [all28.index(c) for c in m.classes_]] = m.predict_proba(Xt)
            probs.append(pr)
        r["k16_pred"], r["k16_prob"] = np.array(preds), np.mean(probs, 0)
        res[key] = r
        print(f"  {key}: done")
    return res


def part_b(res, all28, short):
    rows = []
    for key, r in res.items():
        lab_pred = np.array(r["classes"])[r["pd_prob"].argmax(1)]
        for c in sorted(set(r["pd_y"])):
            m = r["pd_y"] == c
            rows.append({"class": short[c], "label": c, "n_test": int(m.sum()), "model": short_model(key),
                         "lab-trained": 100 * np.mean(lab_pred[m] == c),
                         f"K={K_AUDIT} field": 100 * np.mean(r["k16_pred"][:, m] == c)})
    df = pd.DataFrame(rows)
    df.round(1).to_csv(OUT / "perclass_plantdoc.csv", index=False)

    apply_style()
    models = list(dict.fromkeys(df.model))
    order = (df.groupby("class")[f"K={K_AUDIT} field"].mean().sort_values(ascending=False).index)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.6), sharey=True)
    for ax, col in zip(axes, ["lab-trained", f"K={K_AUDIT} field"]):
        mat = df.pivot(index="class", columns="model", values=col).loc[order, models]
        im = ax.imshow(mat.values, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.5,
                        color="white" if v < 60 else "black")
        ax.set_xticks(range(len(models)), models, rotation=60, ha="right", fontsize=6)
        ax.set_title("Trained on lab images only" if col == "lab-trained"
                     else f"{K_AUDIT} field images per class", fontsize=8)
        ax.spines[:].set_visible(False)
    n = df.drop_duplicates("class").set_index("class").n_test
    axes[0].set_yticks(range(len(order)), [f"{c} (n={n[c]})" for c in order], fontsize=6)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("Per-class accuracy on PlantDoc test (%)", fontsize=7)
    save_figure(fig, "fig_perclass", data=df)
    hard = df.groupby("class")[f"K={K_AUDIT} field"].mean().sort_values().head(5)
    print(f"\nB. Hardest PlantDoc classes (mean over models, K={K_AUDIT}):")
    for c, v in hard.items():
        print(f"   {c:<28} {v:5.1f}%")


def part_c(res, all28, short, key="bioclip2"):
    if key not in res:
        return
    r = res[key]
    cls = r["classes"]
    pred = np.array(cls)[r["pd_prob"].argmax(1)]
    present = sorted(set(r["pd_y"]))
    cm = confusion_matrix(r["pd_y"], pred, labels=all28)
    rows = []
    for i, t in enumerate(all28):
        for j, p in enumerate(all28):
            if i != j and cm[i, j] > 0:
                rows.append({"true": short[t], "predicted": short[p], "count": int(cm[i, j]),
                             "share_of_true_class_%": 100 * cm[i, j] / max(1, cm[i].sum()),
                             "same_crop": t.split("___")[0] == p.split("___")[0]})
    conf = pd.DataFrame(rows).sort_values("count", ascending=False)
    conf.round(1).to_csv(OUT / f"confusions_{key}_labtrained.csv", index=False)
    wrong = cm.sum() - np.trace(cm)
    same = conf[conf.same_crop]["count"].sum()

    apply_style()
    keep = [all28.index(c) for c in present]
    norm = cm[keep][:, keep] / np.maximum(1, cm[keep].sum(1, keepdims=True)) * 100
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    names = [short[c] for c in present]
    ax.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=5.5)
    ax.set_yticks(range(len(names)), names, fontsize=5.5)
    for i in range(len(names)):
        for j in range(len(names)):
            if norm[i, j] >= 10:
                ax.text(j, i, f"{norm[i, j]:.0f}", ha="center", va="center", fontsize=4.5,
                        color="white" if norm[i, j] > 55 else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class (PlantDoc test)")
    ax.set_title(f"{fm.MODELS[key]['label']}, trained on lab images only (row %)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    save_figure(fig, "fig_confusion", data=pd.DataFrame(norm, index=names, columns=names).reset_index())
    print(f"\nC. {fm.MODELS[key]['label']} lab-trained on PlantDoc: {wrong} errors, "
          f"{100 * same / max(1, wrong):.0f}% of them are the right crop but the wrong disease")
    print("   Top 5 confusions:")
    for _, x in conf.head(5).iterrows():
        print(f"   {x.true:<28} -> {x.predicted:<28} {x['count']:>3} photos")


def part_d(res):
    rows = []

    def add(name, kind, prob, y, classes, data):
        conf = prob.max(1)
        correct = (np.array(classes)[prob.argmax(1)] == y).astype(float)
        rows.append({"model": name, "kind": kind, "data": data, "accuracy": 100 * correct.mean(),
                     "mean_confidence": 100 * conf.mean(), "ECE": ece(conf, correct)})

    for key, r in res.items():
        kind = "frozen CNN" if fm.MODELS[key].get("cnn") else "frozen foundation model"
        add(short_model(key), kind, r["lab_prob"], r["lab_y"], r["classes"], "Lab (PlantVillage)")
        add(short_model(key), kind, r["pd_prob"], r["pd_y"], r["classes"], "PlantDoc")
        if "pp_prob" in r:
            add(short_model(key), kind, r["pp_prob"], r["pp_y"], r["classes"], "Plant Pathology")
    all28, _ = fm.class_list()
    for key in fm.CNNS:
        p = fm.ROOT / "results" / "stats" / f"finetuned_{key}_logits.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        for split, data in [("plantvillage_test_sub", "Lab (PlantVillage)"), ("plantdoc_test", "PlantDoc"),
                            ("pp_test", "Plant Pathology")]:
            if f"logits_{split}" not in z.files:
                continue
            lg = z[f"logits_{split}"]
            e = np.exp(lg - lg.max(1, keepdims=True))
            prob = e / e.sum(1, keepdims=True)
            y = np.array(all28)[z[f"y_{split}"]]
            add(short_model(key) + FT_LABEL, "fine-tuned CNN", prob, y, all28, data)
    df = pd.DataFrame(rows)
    df.round(2).to_csv(OUT / "calibration.csv", index=False)

    apply_style()
    datas = ["Lab (PlantVillage)", "PlantDoc", "Plant Pathology"]
    models = list(dict.fromkeys(df.model))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    w = 0.8 / len(datas)
    for i, d in enumerate(datas):
        g = df[df.data == d].set_index("model").reindex(models)
        x = np.arange(len(models)) + (i - 1) * w
        axes[0].bar(x, g.ECE, w, color=COLORS[[2, 0, 1][i]], label=d)
        axes[1].bar(x, g.mean_confidence - g.accuracy, w, color=COLORS[[2, 0, 1][i]], label=d)
    for ax, t in zip(axes, ["Expected calibration error (%, lower = better)",
                            "Overconfidence = mean confidence - accuracy (pts)"]):
        ax.set_xticks(range(len(models)), models, rotation=60, ha="right", fontsize=6)
        ax.set_title(t, fontsize=7.5)
        ax.axhline(0, color="0.3", lw=0.5)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    save_figure(fig, "fig_calibration", data=df)
    print("\nD. Calibration of lab-trained classifiers (mean confidence vs accuracy, %)")
    piv = df.pivot_table(index="model", columns="data", values=["accuracy", "mean_confidence", "ECE"]).round(1)
    print(piv.reindex(models).to_string())


def part_e(res, short):
    found = [k for k in AUDIT_MODELS if k in res]
    if len(found) < AUDIT_MIN_AGREE:
        return
    y, paths = res[found[0]]["pd_y"], res[found[0]]["pd_paths"]
    votes = []
    for k in found:
        assert list(res[k]["pd_paths"]) == list(paths)
        p = res[k]["k16_pred"]                                   # 5 repeats x images
        votes.append([pd.Series(p[:, i]).mode().iloc[0] for i in range(p.shape[1])])
    votes = np.array(votes)                                      # models x images
    rows = []
    for i in range(len(y)):
        vals, cnt = np.unique(votes[:, i], return_counts=True)
        top, n = vals[cnt.argmax()], cnt.max()
        if top != y[i] and n >= AUDIT_MIN_AGREE:
            prob = np.mean([res[k]["k16_prob"][i, list(fm.class_list()[0]).index(top)] for k in found])
            rows.append({"path": paths[i], "given_label": short[y[i]], "models_say": short[top],
                         "models_agreeing": f"{n}/{len(found)}", "mean_probability": round(100 * prob, 1),
                         "your_decision": ""})
    df = pd.DataFrame(rows).sort_values("mean_probability", ascending=False) if rows else pd.DataFrame(
        columns=["path", "given_label", "models_say", "models_agreeing", "mean_probability", "your_decision"])
    df.to_csv(OUT / "label_audit_candidates.csv", index=False)
    print(f"\nE. Label audit: {len(df)} of {len(y)} PlantDoc test photos where >= {AUDIT_MIN_AGREE} of "
          f"{len(found)} foundation models agree on a DIFFERENT class than the given label")
    if df.empty:
        return
    from PIL import Image
    n = min(len(df), 48)
    cols = 6
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(cols * 1.9, rws * 2.15), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (j, r) in zip(axes.flat, enumerate(df.head(n).itertuples())):
        with Image.open(fm.ROOT / r.path) as im:
            im.thumbnail((256, 256))
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"#{j + 1}  given: {r.given_label}\nmodels: {r.models_say} ({r.models_agreeing})",
                     fontsize=5.5)
    fig.tight_layout()
    fig.savefig(OUT / "label_audit_sheet.png", dpi=200)
    plt.close(fig)
    print("   Check them: results/analysis/label_audit_sheet.png (numbers = row order in "
          "label_audit_candidates.csv). Fill 'your_decision' with: correct / wrong / unsure")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all28, short = fm.class_list()
    part_a()
    print("\nRecomputing per-image predictions ...")
    res = collect(all28)
    part_b(res, all28, short)
    part_c(res, all28, short)
    part_d(res)
    part_e(res, short)
    print("\nSaved: figures/fig_lab_vs_field.*, fig_perclass.*, fig_confusion.*, fig_calibration.* "
          "and results/analysis/")


if __name__ == "__main__":
    main()
