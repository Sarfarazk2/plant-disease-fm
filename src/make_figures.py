"""
Step 18 - FINAL paper figures and tables, all in one consistent style.
Running time: about 2-5 minutes (reads the saved results; no models, no GPU).

Run from the project folder (plant-disease-fm), after all other steps:
    python src/make_figures.py

Every figure is saved in figures/final/ as SVG, PDF, EPS (editable, real text), PNG and TIFF (600 dpi)
plus a CSV of the exact plotted numbers. Every table is saved in results/paper_tables/ as CSV and
LaTeX, and all tables go into ONE Excel workbook (results/paper_tables/paper_tables.xlsx) with
native, editable Excel charts for the key results (needs:  pip install openpyxl).

Main figures
  Fig 1  Study design (schematic)
  Fig 2  Lab accuracy vs field accuracy, 13 models (slope chart, 95% CIs)
  Fig 3  Size-matched control: biology training vs the same general model (two pairs)
  Fig 4  Few-shot curves on the three field datasets
  Fig 5  Fusion v2: loss against the better single source in every case
  Fig 6  Unknown-crop and unknown-disease detection
Supplementary figures
  Fig S1 Data efficiency: 16 photos per class vs all PlantDoc training photos
  Fig S2 Prompt-template sensitivity of zero-shot accuracy
  Fig S3 Calibration in the field before / after temperature scaling fitted on lab images
  (fig_perclass, fig_confusion, fig_calibration from analysis.py are kept as further supplements)
Tables
  T1 datasets, T2 models, T3 lab-to-field, T4 zero-shot, T5 few-shot, T6 size-matched control,
  T7 fusion, T8 open-set, S1 full-data, S2 prompt templates, S3 calibration, S4 label audit
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import figstyle  # noqa: E402
import fm  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
R = fm.ROOT / "results"
FIG = fm.ROOT / "figures" / "final"
TAB = R / "paper_tables"
figstyle.FIG_DIR = FIG

# ---------------------------------------------------------------- one style for every model
STYLE = {   # key: (short label, colour, marker)
    "bioclip2":     ("BioCLIP 2", "#009E73", "o"),
    "bioclip":      ("BioCLIP", "#7FCDB8", "o"),
    "clip_l14":     ("CLIP-L/14 (LAION)", "#0072B2", "s"),
    "clip_b16":     ("CLIP-B/16 (OpenAI)", "#56B4E9", "s"),
    "siglip_b16":   ("SigLIP-B/16", "#CC79A7", "D"),
    "dinov2_l14":   ("DINOv2-L/14", "#D55E00", "^"),
    "dinov2_b14":   ("DINOv2-B/14", "#E69F00", "^"),
    "resnet50":     ("ResNet-50", "#4D4D4D", "v"),
    "effnet_b0":    ("EfficientNet-B0", "#808080", "v"),
    "mobilenet_v3": ("MobileNetV3-L", "#B3B3B3", "v"),
}
FAMILY = {"bioclip2": "Biology vision-language", "bioclip": "Biology vision-language",
          "clip_l14": "Web vision-language", "clip_b16": "Web vision-language", "siglip_b16": "Web vision-language",
          "dinov2_l14": "Self-supervised", "dinov2_b14": "Self-supervised",
          "resnet50": "ImageNet CNN", "effnet_b0": "ImageNet CNN", "mobilenet_v3": "ImageNet CNN"}
LABEL2KEY = {v["label"]: k for k, v in fm.MODELS.items()}
LABEL2KEY.update({v["label"].replace(" (ImageNet)", ""): k for k, v in fm.MODELS.items()})


def key_of(name):
    """CSV model name -> (key, fine_tuned?)."""
    ft = str(name).endswith(" (fine-tuned)")
    base = str(name).replace(" (fine-tuned)", "")
    return LABEL2KEY.get(base, base if base in fm.MODELS else None), ft


def short(key, ft=False):
    return STYLE[key][0] + (" fine-tuned" if ft else "")


def read(path):
    p = R / path
    if not p.exists():
        print(f"  (missing {path} - parts that need it are skipped)")
        return None
    return pd.read_csv(p)


def spread(y, gap):
    y = np.asarray(y, float)
    order = np.argsort(y)
    ys = y[order]
    clusters = [[i] for i in range(len(ys))]

    def place(c):
        return ys[c].mean() - (len(c) - 1) / 2 * gap + gap * np.arange(len(c))

    merged = True
    while merged:
        merged = False
        for j in range(1, len(clusters)):
            if place(clusters[j - 1])[-1] + gap > place(clusters[j])[0]:
                clusters[j - 1] += clusters.pop(j)
                merged = True
                break
    out = np.empty(len(y))
    out[order] = np.concatenate([place(c) for c in clusters])
    return out


def tint(color, a):
    """Solid colour mixed with white (instead of transparency, which EPS cannot store)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (1 - a * (1 - r), 1 - a * (1 - g), 1 - a * (1 - b))


def ci_lookup(acc, table, dataset, method, k=None):
    if acc is None:
        return {}
    g = acc[(acc.table == table) & (acc.dataset == dataset) & (acc.method == method)]
    if k is not None:
        g = g[g.K == k]
    return {(key_of(r.model)[0]): (r.accuracy, r.ci95_low, r.ci95_high) for r in g.itertuples()}


# ================================================================ FIGURES
def fig1():
    figstyle.apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    def box(x, w, title, lines, color):
        y, h = 7, 21
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=color, ec="0.35", lw=0.7))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7.2, weight="bold")
        ax.text(x + w / 2, y + h - 5.4, "\n".join(lines), ha="center", va="top", fontsize=6.1, linespacing=1.4)

    def arrow(x0, x1):
        ax.add_patch(FancyArrowPatch((x0, 17.5), (x1, 17.5), arrowstyle="-|>", mutation_scale=9, lw=0.8,
                                     color="0.3"))

    box(0.8, 19.5, "Lab training data", ["PlantVillage", "28 classes shared", "with PlantDoc",
                                         "5,600 train / 2,669 test", "(leaf-grouped split)"], "#F2F2F2")
    box(23.3, 25.5, "13 image encoders", ["Biology vision-language:", "BioCLIP, BioCLIP 2",
                                          "Web vision-language: CLIP-B,", "CLIP-L, SigLIP \u00b7 Self-supervised:",
                                          "DINOv2-B, -L \u00b7 ImageNet CNNs (3,", "frozen and fully fine-tuned)"],
        "#E8F4EF")
    box(52.0, 22.0, "Ways to classify", ["Zero-shot text prompts", "Lab-trained linear classifier",
                                         "Few-shot: K = 1, 5, 10, 16", "field photos per class",
                                         "v2 text + image fusion"], "#EAF2FA")
    box(77.2, 22.0, "Field tests", ["PlantDoc (web) \u00b7 236, 28-way", "Plant Pathology 2020",
                                    "(apple orchard) \u00b7 1,580", "JIC wheat (new crop)",
                                    "849, 5-way"], "#FDF1E6")
    for x0, x1 in [(20.9, 22.7), (49.4, 51.4), (74.6, 76.6)]:
        arrow(x0, x1)
    ax.text(50, 2.6, "No setting is tuned on field data \u00b7 Checks: size-matched control (same architecture, "
            "with vs without biology training), unknown-disease detection, calibration, PlantDoc label audit, "
            "bootstrap CIs and permutation tests", ha="center", va="center", fontsize=5.9, style="italic",
            color="0.25", wrap=True)
    figstyle.save_figure(fig, "Fig1_study_design")


def fig2(lvf, acc):
    if lvf is None:
        return
    figstyle.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
    rows = []
    for ax, field in zip(axes, ["PlantDoc", "Plant Pathology"]):
        cis = {**{(k, False): v for k, v in ci_lookup(acc, "lab-trained", field, "frozen + linear").items()},
               **{(k, True): v for k, v in ci_lookup(acc, "lab-trained", field, "fully fine-tuned").items()}}
        d = lvf.dropna(subset=[field]).copy()
        d["key"] = d.model.map(lambda m: key_of(m)[0])
        d["ft"] = d.model.map(lambda m: key_of(m)[1])
        d = d[d.key.notna()]
        for _, r in d.iterrows():
            c, mk = STYLE[r.key][1], STYLE[r.key][2]
            val = r[field]
            ax.plot([0, 1], [r.lab, val], color=c, ls="--" if r.ft else "-", lw=1.1 if r.key == "bioclip2" else 0.8,
                    marker=mk, ms=3.5, mfc="white" if r.ft else c, zorder=3 if r.key == "bioclip2" else 2)
            lo_hi = cis.get((r.key, r.ft))
            if lo_hi:
                ax.plot([1.0, 1.0], [lo_hi[1], lo_hi[2]], color=tint(c, 0.6), lw=0.6)
            rows.append({"field_test": field, "model": short(r.key, r.ft), "family": FAMILY[r.key],
                         "fine_tuned": r.ft, "lab_accuracy": r.lab, "field_accuracy": val,
                         "field_ci95_low": lo_hi[1] if lo_hi else None, "field_ci95_high": lo_hi[2] if lo_hi else None})
        vals = d[field].values
        ly = spread(vals, 2.9)
        for (_, r), y, v in zip(d.iterrows(), ly, vals):
            ax.annotate(short(r.key, r.ft), xy=(1.02, v), xytext=(1.1, y), fontsize=5.8, va="center",
                        color="black" if r.key == "bioclip2" else "0.25",
                        weight="bold" if r.key == "bioclip2" else "normal",
                        arrowprops=dict(arrowstyle="-", lw=0.3, color="0.65"))
        from scipy.stats import spearmanr
        rho, p = spearmanr(d.lab, d[field])
        ax.set_xticks([0, 1], ["Lab test\n(PlantVillage)", f"Field test\n({field})"])
        ax.set_xlim(-0.12, 1.85)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"Lab spread {d.lab.max() - d.lab.min():.1f} pts → field spread "
                     f"{d[field].max() - d[field].min():.1f} pts\nrank correlation lab vs field "
                     f"ρ = {rho:.2f} (p = {p:.2f}, n = {len(d)})", fontsize=7)
    axes[0].set_ylabel("Accuracy (%), classifier trained on lab images only")
    fams = ["Biology vision-language", "Web vision-language", "Self-supervised", "ImageNet CNN"]
    rep = {"Biology vision-language": "bioclip2", "Web vision-language": "clip_l14",
           "Self-supervised": "dinov2_b14", "ImageNet CNN": "resnet50"}
    h = [plt.Line2D([], [], color=STYLE[rep[f]][1], marker=STYLE[rep[f]][2], ms=4, label=f) for f in fams]
    h.append(plt.Line2D([], [], color="0.4", ls="--", marker="v", mfc="white", ms=4, label="fully fine-tuned CNN"))
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    figstyle.save_figure(fig, "Fig2_lab_vs_field", data=pd.DataFrame(rows))


TESTS = [("lab-trained", "PlantDoc", None, "frozen + linear", "Lab-trained → PlantDoc"),
         ("lab-trained", "Plant Pathology", None, "frozen + linear", "Lab-trained → Plant Pathology"),
         ("zero-shot", "PlantDoc", 0, "zero-shot text", "Zero-shot, PlantDoc"),
         ("zero-shot", "Plant Pathology", 0, "zero-shot text", "Zero-shot, Plant Pathology"),
         ("zero-shot", "JIC wheat", 0, "zero-shot text", "Zero-shot, wheat"),
         ("few-shot", "PlantDoc", 16, "image only", "16-shot, PlantDoc"),
         ("few-shot", "Plant Pathology", 16, "image only", "16-shot, apple (3-way)"),
         ("few-shot", "JIC wheat", 16, "image only", "16-shot, wheat (5-way)")]


def size_control_rows(acc, pairs):
    rows = []
    for bio, base in [("bioclip", "clip_b16"), ("bioclip2", "clip_l14")]:
        for table, ds, k, meth, lab in TESTS:
            a = ci_lookup(acc, table, ds, meth, k)
            if bio not in a or base not in a:
                continue
            sig, diff_ci = "", (None, None)
            if pairs is not None:
                g = pairs[(pairs.pair == f"{fm.MODELS[bio]['label']} vs {fm.MODELS[base]['label']}")
                          & (pairs.table == table) & (pairs.dataset == ds) & (pairs.method == meth)]
                if k is not None:
                    g = g[g.K == k]
                if not g.empty:
                    sig, diff_ci = g.sig.iloc[0], (g.diff_ci95_low.iloc[0], g.diff_ci95_high.iloc[0])
            rows.append({"pair": f"{short(base)} → {short(bio)}", "test": lab,
                         "general_model": short(base), "general_acc": a[base][0], "general_lo": a[base][1],
                         "general_hi": a[base][2], "biology_model": short(bio), "biology_acc": a[bio][0],
                         "biology_lo": a[bio][1], "biology_hi": a[bio][2], "difference": a[bio][0] - a[base][0],
                         "diff_ci95_low": diff_ci[0], "diff_ci95_high": diff_ci[1], "significance": sig})
    return pd.DataFrame(rows)


def fig3(acc, pairs):
    df = size_control_rows(acc, pairs)
    if df.empty:
        return df
    figstyle.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    tests = [t[4] for t in TESTS]
    for ax, (pair, g) in zip(axes, df.groupby("pair", sort=False)):
        bio_key = "bioclip2" if "BioCLIP 2" in pair else "bioclip"
        base_key = "clip_l14" if bio_key == "bioclip2" else "clip_b16"
        for r in g.itertuples():
            y = len(tests) - 1 - tests.index(r.test)
            ax.plot([r.general_acc, r.biology_acc], [y, y], color="0.75", lw=1.6, zorder=1)
            for x, lo, hi, kk in [(r.general_acc, r.general_lo, r.general_hi, base_key),
                                  (r.biology_acc, r.biology_lo, r.biology_hi, bio_key)]:
                ax.plot([lo, hi], [y, y], color=tint(STYLE[kk][1], 0.6), lw=0.8, zorder=2)
                ax.plot(x, y, marker=STYLE[kk][2], color=STYLE[kk][1], ms=5, zorder=3)
            txt = f"{r.difference:+.1f}" + (f" {r.significance}" if r.significance and r.significance != "n.s." else
                                             (" n.s." if r.significance == "n.s." else ""))
            ax.text(101, y, txt, va="center", fontsize=6.2, color="#00664A" if r.difference > 0 else "#00507D")
        ax.set_xlim(15, 100)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(f"{STYLE[base_key][0]} → {STYLE[bio_key][0]}\n(same architecture and starting weights)",
                     fontsize=7.2)
        ax.grid(axis="x", lw=0.3, color="0.9")
    axes[0].set_yticks(range(len(tests)), list(reversed(tests)), fontsize=6.5)
    fig.legend(handles=[plt.Line2D([], [], color=STYLE[k][1], marker=STYLE[k][2], ls="", ms=5, label=STYLE[k][0])
                        for k in ["clip_b16", "bioclip", "clip_l14", "bioclip2"]],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.035), fontsize=6.3)
    fig.text(0.5, 0.005, "Numbers at right: biology model minus general model (points); * p<0.05, ** p<0.01, "
             "*** p<0.001, paired permutation test, Holm-corrected. Whiskers: 95% bootstrap CI.",
             ha="center", fontsize=5.5, color="0.3")
    fig.tight_layout(rect=(0, 0.1, 0.97, 1))
    figstyle.save_figure(fig, "Fig3_size_matched_control", data=df)
    return df


FS_MODELS = ["bioclip2", "bioclip", "clip_l14", "siglip_b16", "dinov2_b14", "mobilenet_v3"]


def fig4(pd_s, pp_s, wh_s):
    figstyle.apply_style()
    panels = [("PlantDoc (28-way)", pd_s, "linear probe", "zero-shot text"),
              ("Plant Pathology, apple (3-way)", pp_s, "linear probe", "zero-shot text"),
              ("JIC wheat, new crop (5-way)", wh_s, "image only", "text only")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    xpos = {0: 0, 1: 1, 5: 2, 10: 3, 16: 4}
    rows = []
    for ax, (title, s, meth, zs) in zip(axes, panels):
        if s is None:
            ax.set_visible(False)
            continue
        for m in FS_MODELS:
            g = s[(s.model == m) & (s.method == meth)].sort_values("K")
            if g.empty:
                continue
            c, mk = STYLE[m][1], STYLE[m][2]
            x = [xpos[k] for k in g.K]
            ax.plot(x, g.accuracy_mean, color=c, marker=mk, ms=3, lw=1.3 if m == "bioclip2" else 0.9,
                    label=STYLE[m][0], zorder=3 if m == "bioclip2" else 2)
            ax.fill_between(x, g.accuracy_mean - g.accuracy_std, g.accuracy_mean + g.accuracy_std,
                            color=tint(c, 0.15), lw=0, zorder=1)
            z = s[(s.model == m) & (s.method == zs) & (s.K == 0)]
            if not z.empty:
                ax.plot(0, z.accuracy_mean.iloc[0], marker=mk, mfc="white", color=c, ms=4, ls="")
                ax.plot([0, 1], [z.accuracy_mean.iloc[0], g.accuracy_mean.iloc[0]], color=c, lw=0.5, ls=":")
                rows.append({"dataset": title, "model": STYLE[m][0], "K": 0, "method": "zero-shot text",
                             "accuracy_mean": z.accuracy_mean.iloc[0], "accuracy_std": 0})
            rows += [{"dataset": title, "model": STYLE[m][0], "K": int(r.K), "method": "linear probe",
                      "accuracy_mean": r.accuracy_mean, "accuracy_std": r.accuracy_std} for r in g.itertuples()]
        ax.set_xticks(list(xpos.values()), ["0\n(text)", "1", "5", "10", "16"])
        ax.set_xlabel("Field photos per class (K)")
        ax.set_title(title, fontsize=7.2)
        ax.grid(axis="y", lw=0.3, color="0.9")
    axes[0].set_ylabel("Accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="0.4", marker="o", mfc="white", ls="", label="zero-shot (K = 0)"))
    fig.legend(handles=h, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.05), fontsize=6.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    figstyle.save_figure(fig, "Fig4_fewshot_curves", data=pd.DataFrame(rows))


def fusion_cells(cells):
    if cells is None:
        return None
    piv = cells.pivot_table(index=["dataset", "model", "K"], columns="baseline", values="baseline_acc")
    piv["reliability v2 (ours)"] = cells.groupby(["dataset", "model", "K"]).v2_acc.first()
    oracle = piv[["text only", "image only"]].max(axis=1)
    loss = pd.DataFrame({m: oracle - piv[m] for m in ["text only", "image only", "equal fusion",
                                                       "reliability v2 (ours)"]})
    return loss.reset_index()


def fig5(cells):
    loss = fusion_cells(cells)
    if loss is None:
        return
    figstyle.apply_style()
    methods = ["text only", "image only", "equal fusion", "reliability v2 (ours)"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        v = loss[m].values
        c = STYLE["bioclip2"][1] if "v2" in m else "0.45"
        ax.boxplot(v, positions=[i], orientation="horizontal", widths=0.5, showfliers=False,
                   medianprops=dict(color=c, lw=1.2), boxprops=dict(color=c, lw=0.8),
                   whiskerprops=dict(color=c, lw=0.8), capprops=dict(color=c, lw=0.8))
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=5, color=tint(c, 0.6), lw=0, zorder=3)
        ax.text(max(loss[methods].max()) + 1.5, i, f"mean {v.mean():.1f}\nworst {v.max():.1f}",
                va="center", fontsize=5.8, color=c)
    ax.axvline(0, color="0.2", lw=0.6)
    ax.set_yticks(range(len(methods)), methods, fontsize=6.8)
    ax.set_xlabel("Loss vs the better of text-only and image-only\n(points; negative = better than both)")
    ax.set_title(f"All {len(loss)} cases: {loss.dataset.nunique()} datasets × {loss.model.nunique()} "
                 f"models × K = 1, 5, 10, 16", fontsize=6.8)
    ax.set_xlim(right=max(loss[methods].max()) + 12)
    fig.tight_layout()
    figstyle.save_figure(fig, "Fig5_fusion_v2_loss", data=loss)


def fig6(osum):
    if osum is None:
        return
    figstyle.apply_style()
    g = osum[osum.method == "prototype"].copy()
    far = g[g.setting.str.startswith("far")].set_index("model")
    near = g[g.setting.str.startswith("near")].set_index("model")
    models = far.auroc_mean.sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    rows = []
    for i, m in enumerate(models):
        for d, mk, lab in [(far, "o", "unknown crop (wheat photos)"), (near, "s", "unknown disease (7 hidden PlantDoc classes)")]:
            if m not in d.index:
                continue
            ax.errorbar(d.loc[m, "auroc_mean"], i, xerr=d.loc[m, "auroc_std"], fmt=mk, color=STYLE[m][1],
                        mfc=STYLE[m][1] if mk == "o" else "white", ms=4, elinewidth=0.6, capsize=1.5)
            rows.append({"model": STYLE[m][0], "setting": lab, "auroc_mean": d.loc[m, "auroc_mean"],
                         "auroc_std": d.loc[m, "auroc_std"], "fpr95_mean": d.loc[m, "fpr95_mean"]})
    ax.axvline(50, color="0.5", lw=0.6, ls=":")
    ax.text(50.5, -0.75, "random", fontsize=5.5, color="0.4", va="center")
    ax.set_ylim(-1.1, len(models) - 0.5)
    ax.set_yticks(range(len(models)), [STYLE[m][0] for m in models], fontsize=6.5)
    ax.set_xlabel("AUROC (%), prototype similarity, K = 16 (mean \u00b1 SD, 5 repeats)")
    ax.legend(handles=[plt.Line2D([], [], marker="o", color="0.35", ls="", ms=4, label="unknown crop (wheat photos)"),
                       plt.Line2D([], [], marker="s", color="0.35", mfc="white", ls="", ms=4,
                                  label="unknown disease (7 hidden PlantDoc classes)")],
              fontsize=5.8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, frameon=False)
    ax.set_xlim(45, 100)
    fig.tight_layout()
    figstyle.save_figure(fig, "Fig6_unknown_detection", data=pd.DataFrame(rows))


def figS1(full):
    if full is None:
        return
    figstyle.apply_style()
    o = full[full.test_version == "original"]
    piv = o.pivot_table(index="model", columns="training_pool", values="accuracy")
    if "K=16 per class (reference)" not in piv:
        return
    piv = piv.dropna().sort_values("all training photos")
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for i, (mname, r) in enumerate(piv.iterrows()):
        k = key_of(mname)[0]
        c = STYLE[k][1]
        ax.plot([r["K=16 per class (reference)"], r["all training photos"]], [i, i], color="0.8", lw=1.5)
        ax.plot(r["K=16 per class (reference)"], i, "o", mfc="white", color=c, ms=4)
        ax.plot(r["all training photos"], i, "o", color=c, ms=4)
        ax.text(r["all training photos"] + 0.8, i,
                f"{100 * r['K=16 per class (reference)'] / r['all training photos']:.0f}%", va="center", fontsize=5.8)
    ax.set_yticks(range(len(piv)), [STYLE[key_of(m)[0]][0] for m in piv.index], fontsize=6.5)
    ax.set_xlabel("PlantDoc accuracy (%), frozen features + linear classifier")
    ax.legend(handles=[plt.Line2D([], [], marker="o", mfc="white", color="0.4", ls="", label="16 photos per class"),
                       plt.Line2D([], [], marker="o", color="0.4", ls="", label="all 2,313 training photos")],
              fontsize=6, loc="upper left")
    fig.tight_layout()
    figstyle.save_figure(fig, "FigS1_data_efficiency", data=piv.reset_index())


def figS2(pt):
    if pt is None:
        return
    figstyle.apply_style()
    datas = ["PlantDoc 28-way", "Plant Pathology 28-way", "Wheat 5-way"]
    models = [m for m in ["bioclip2", "bioclip", "clip_l14", "clip_b16", "siglip_b16"]
              if fm.MODELS[m]["label"] in set(pt.model)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, d in zip(axes, datas):
        for i, m in enumerate(models):
            g = pt[(pt.model == fm.MODELS[m]["label"]) & (pt.data == d)]
            single = g[g.template != "average of 4 (used)"].accuracy
            avg = g[g.template == "average of 4 (used)"].accuracy
            if single.empty:
                continue
            ax.plot([single.min(), single.max()], [i, i], color=tint(STYLE[m][1], 0.35), lw=3, solid_capstyle="butt")
            ax.plot(single, [i] * len(single), "|", color=STYLE[m][1], ms=6)
            if not avg.empty:
                ax.plot(avg.iloc[0], i, marker="D", color=STYLE[m][1], ms=4)
        ax.set_title(d, fontsize=7)
        ax.set_xlabel("Zero-shot accuracy (%)")
        ax.grid(axis="x", lw=0.3, color="0.9")
    axes[0].set_yticks(range(len(models)), [STYLE[m][0] for m in models], fontsize=6.5)
    fig.legend(handles=[plt.Line2D([], [], marker="|", color="0.4", ls="", ms=6, label="single template"),
                        plt.Line2D([], [], marker="D", color="0.4", ls="", ms=4, label="average of 4 templates (used)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), fontsize=6.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    figstyle.save_figure(fig, "FigS2_prompt_templates", data=pt)


def figS3(ct):
    if ct is None:
        return
    figstyle.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    names = list(dict.fromkeys(ct.model))
    for ax, d in zip(axes, ["PlantDoc", "Plant Pathology"]):
        for i, mname in enumerate(names):
            k, ft = key_of(mname)
            g = ct[(ct.model == mname) & (ct.data == d)].set_index("scaling")
            if g.empty or k is None:
                continue
            b, a = g.loc["before", "ECE"], g.loc["after", "ECE"]
            ax.annotate("", xy=(a, i), xytext=(b, i), arrowprops=dict(arrowstyle="-|>", color=STYLE[k][1], lw=0.9))
            ax.plot(b, i, "o", color=STYLE[k][1], mfc="white" if ft else STYLE[k][1], ms=3.5)
        ax.set_title(f"{d}: ECE before → after lab-fitted temperature scaling", fontsize=6.8)
        ax.set_xlabel("Expected calibration error (%), lower = better")
        ax.grid(axis="x", lw=0.3, color="0.9")
    axes[0].set_yticks(range(len(names)), [short(*key_of(n)) if key_of(n)[0] else n for n in names], fontsize=6)
    fig.tight_layout()
    figstyle.save_figure(fig, "FigS3_calibration_transfer", data=ct)


# ================================================================ TABLES
def fmt_ci(v):
    return "" if v is None else f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"


def tables(lvf, acc, pairs_df, pd_s, pp_s, wh_s, pps, cells, osum, full, pt, ct, cost, resc, audit, fused_over):
    TAB.mkdir(parents=True, exist_ok=True)
    out = {}
    # T1 datasets
    def n(split):
        p = fm.SPLITS / f"{split}.csv"
        return len(fm.read_split(split)) if p.exists() else None
    out["T1_datasets"] = pd.DataFrame([
        {"dataset": "PlantVillage", "role": "lab training source", "photo type": "lab, plain background",
         "classes": 28, "train / pool": n("plantvillage_train_sub"), "test": n("plantvillage_test_sub")},
        {"dataset": "PlantDoc", "role": "field test 1 + few-shot pool", "photo type": "web field photos",
         "classes": 28, "train / pool": n("plantdoc_train_clean"), "test": n("plantdoc_test")},
        {"dataset": "Plant Pathology 2020", "role": "field test 2 (non-web)", "photo type": "apple orchard",
         "classes": 3, "train / pool": n("pp_pool"), "test": n("pp_test")},
        {"dataset": "JIC wheat", "role": "new-crop test + unknown-crop set", "photo type": "wheat field",
         "classes": 5, "train / pool": n("wheat_pool"), "test": n("wheat_test")}])
    # T2 models
    rows = []
    for k, v in fm.MODELS.items():
        r = {"model": v["label"], "family": FAMILY.get(k, ""), "text encoder": "yes" if fm.has_text_encoder(k) else "no",
             "prompt style": v.get("prompt", "")}
        if cost is not None:
            c = cost[cost.model == v["label"]]
            if not c.empty:
                r["image encoder params (M)"] = c.image_encoder_params_M.iloc[0]
                r["images/s (laptop GPU)"] = c.images_per_second_gpu.iloc[0]
        rows.append(r)
    out["T2_models"] = pd.DataFrame(rows)
    # T3 lab-to-field
    if lvf is not None:
        rows = []
        for r in lvf.itertuples():
            k, ft = key_of(r.model)
            if k is None:
                continue
            meth = "fully fine-tuned" if ft else "frozen + linear"
            row = {"model": short(k, ft), "lab accuracy": round(r.lab, 1)}
            for field in ["PlantDoc", "Plant Pathology"]:
                row[f"{field} [95% CI]"] = fmt_ci(ci_lookup(acc, "lab-trained", field, meth).get(k))
            if resc is not None:
                m = "lab-trained (fully fine-tuned)" if ft else "lab-trained (frozen + linear)"
                g = resc[(resc.method == m) & (resc.model == (fm.MODELS[k]["label"] + (" (fine-tuned)" if ft else "")))
                         & (resc.test_version == "corrected")]
                row["PlantDoc, corrected labels"] = round(g.accuracy.iloc[0], 1) if not g.empty else ""
            rows.append(row)
        out["T3_lab_to_field"] = pd.DataFrame(rows)
    # T4 zero-shot
    if acc is not None:
        rows = []
        for k in fm.MODELS:
            if not fm.has_text_encoder(k):
                continue
            row = {"model": short(k), "prompt style": fm.MODELS[k]["prompt"]}
            for ds, lab in [("PlantDoc", "PlantDoc 28-way"), ("Plant Pathology", "Plant Pathology 28-way"),
                            ("JIC wheat", "wheat 5-way")]:
                row[f"{lab} [95% CI]"] = fmt_ci(ci_lookup(acc, "zero-shot", ds, "zero-shot text", 0).get(k))
            if pps is not None:
                g = pps[(pps.model == k) & (pps.setting == "3-way") & pps.method.str.contains("pre-specified")]
                row["Plant Pathology 3-way"] = round(g.accuracy.iloc[0], 1) if not g.empty else ""
            rows.append(row)
        out["T4_zero_shot"] = pd.DataFrame(rows)
    # T5 few-shot (linear probe, mean +/- std over 5 repeats)
    rows = []
    for k in fm.MODELS:
        row = {"model": short(k)}
        for s, meth, lab in [(pd_s, "linear probe", "PlantDoc"), (pp_s, "linear probe", "apple"),
                             (wh_s, "image only", "wheat")]:
            if s is None:
                continue
            for K in [1, 5, 16]:
                g = s[(s.model == k) & (s.method == meth) & (s.K == K)]
                row[f"{lab} K={K}"] = f"{g.accuracy_mean.iloc[0]:.1f} ± {g.accuracy_std.iloc[0]:.1f}" if not g.empty else ""
        rows.append(row)
    out["T5_few_shot"] = pd.DataFrame(rows)
    # T6 size control
    if pairs_df is not None and not pairs_df.empty:
        t = pairs_df[["pair", "test", "general_acc", "biology_acc", "difference", "diff_ci95_low", "diff_ci95_high",
                      "significance"]].copy()
        out["T6_size_matched_control"] = t.round(1)
    # T7 fusion
    if fused_over is not None:
        out["T7_fusion_overall"] = fused_over.round(3)
    # T8 open-set
    if osum is not None:
        t = osum.copy()
        t["model"] = t.model.map(lambda k: STYLE.get(k, (k,))[0])
        out["T8_open_set"] = t.round(1)
    if full is not None:
        out["S1_full_data_plantdoc"] = full.round(1)
    if pt is not None:
        out["S2_prompt_templates"] = pt.round(1)
    if ct is not None:
        out["S3_calibration_temperature"] = ct.round(2)
    if audit is not None:
        out["S4_label_audit_counts"] = audit.groupby(["split", "category"]).size().reset_index(name="images")
    for name, df in out.items():
        figstyle.save_table(df, f"paper_tables/{name}")
    return out


def excel(out, lvf):
    try:
        import openpyxl  # noqa: F401
        from openpyxl.chart import BarChart, Reference
    except ImportError:
        print("\n  Excel workbook skipped: run  pip install openpyxl  and then this script again.")
        return
    path = TAB / "paper_tables.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in out.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
        if lvf is not None:
            d = lvf[["model", "lab", "PlantDoc", "Plant Pathology"]].copy()
            d.columns = ["Model", "Lab test", "PlantDoc (field)", "Plant Pathology (field)"]
            d.to_excel(xw, sheet_name="Chart_lab_vs_field", index=False)
            ws = xw.sheets["Chart_lab_vs_field"]
            ch = BarChart()
            ch.type = "col"
            ch.title = "Lab vs field accuracy (classifier trained on lab images)"
            ch.y_axis.title = "Accuracy (%)"
            ch.add_data(Reference(ws, min_col=2, max_col=4, min_row=1, max_row=len(d) + 1), titles_from_data=True)
            ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=len(d) + 1))
            ch.width, ch.height = 26, 12
            ws.add_chart(ch, "F2")
        if "T6_size_matched_control" in out:
            d = out["T6_size_matched_control"][["pair", "test", "general_acc", "biology_acc"]].copy()
            d.to_excel(xw, sheet_name="Chart_size_control", index=False)
            ws = xw.sheets["Chart_size_control"]
            ch = BarChart()
            ch.type = "bar"
            ch.title = "Biology training vs the same general model"
            ch.x_axis.title = "Test"
            ch.y_axis.title = "Accuracy (%)"
            ch.add_data(Reference(ws, min_col=3, max_col=4, min_row=1, max_row=len(d) + 1), titles_from_data=True)
            ch.set_categories(Reference(ws, min_col=1, max_col=2, min_row=2, max_row=len(d) + 1))
            ch.width, ch.height = 24, 14
            ws.add_chart(ch, "G2")
    print(f"Saved Excel workbook with native charts: results/paper_tables/paper_tables.xlsx")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    lvf = read("analysis/lab_vs_field.csv")
    acc = read("stats/accuracy_ci.csv")
    pairs = read("stats/controlled_pairs.csv")
    pd_s, pp_s, wh_s = read("fewshot/fewshot_summary.csv"), read("plantpathology/pp_fewshot_summary.csv"), \
        read("wheat/wheat_summary.csv")
    pps = read("plantpathology/pp_summary.csv")
    cells, fused_over = read("stats/fusion_v2_tests.csv"), read("stats/fusion_v2_overall.csv")
    osum = read("openset/openset_summary.csv")
    full, pt, ct, cost = read("extras/full_data_plantdoc.csv"), read("extras/prompt_templates.csv"), \
        read("extras/calibration_temperature.csv"), read("extras/model_cost.csv")
    resc, audit = read("analysis/rescore_plantdoc.csv"), read("analysis/plantdoc_label_audit.csv")

    fig1()
    fig2(lvf, acc)
    pairs_df = fig3(acc, pairs)
    fig4(pd_s, pp_s, wh_s)
    fig5(cells)
    fig6(osum)
    figS1(full)
    figS2(pt)
    figS3(ct)
    out = tables(lvf, acc, pairs_df, pd_s, pp_s, wh_s, pps, cells, osum, full, pt, ct, cost, resc, audit, fused_over)
    excel(out, lvf)
    if pairs is None:
        print("\n  Note: run the updated stats.py once to add significance stars to Fig 3 / Table T6.")
    print("\nDone: figures/final/ and results/paper_tables/")


if __name__ == "__main__":
    main()
