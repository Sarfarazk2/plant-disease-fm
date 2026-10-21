"""
figstyle.py - one shared style and save function for ALL paper figures.

Every figure is saved in several formats at once, so it fits any journal:
    figures/<name>.svg   editable vector (text stays editable) - open in Inkscape, Illustrator, PowerPoint, Word
    figures/<name>.pdf   editable vector (fonts embedded as TrueType, text selectable/editable)
    figures/<name>.eps   vector, for journals that ask for EPS
    figures/<name>.png   600 dpi image, for drafts and quick viewing
    figures/<name>.tiff  600 dpi image (LZW-compressed), for journals that ask for TIFF
    figures/<name>_data.csv   the exact numbers plotted, so any chart can be rebuilt or edited in Excel

Usage in any script:
    from figstyle import apply_style, save_figure, COLORS
    apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))   # 3.5 in = single journal column, 7.2 in = full width
    ...
    save_figure(fig, "fig3_fewshot_curves", data=df)   # df = pandas DataFrame with the plotted values
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files without opening windows
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

# Colour-blind-safe palette (Okabe-Ito), widely recommended for scientific figures.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
          "#CC79A7", "#56B4E9", "#F0E442", "#000000", "#999999"]

FORMATS = ["svg", "pdf", "eps", "png", "tiff"]


def apply_style():
    """Journal-ready defaults: readable fonts, editable text in vector files."""
    plt.rcParams.update({
        # Keep text as real, editable text (not outlines) in vector files
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Fonts most journals accept
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # Clean look
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "axes.prop_cycle": matplotlib.cycler(color=COLORS),
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save_figure(fig, name, data=None, formats=FORMATS, dpi=600):
    """Save one figure in every format, plus its data as CSV. Returns the list of files."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = FIG_DIR / f"{name}.{fmt}"
        kwargs = {}
        if fmt in ("png", "tiff"):
            kwargs["dpi"] = dpi
        if fmt == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        if fmt == "eps":
            # EPS cannot store transparency; use a white background.
            kwargs["facecolor"] = "white"
        fig.savefig(path, format=fmt, **kwargs)
        saved.append(path)
    if data is not None:
        path = FIG_DIR / f"{name}_data.csv"
        data.to_csv(path, index=False)
        saved.append(path)
    plt.close(fig)
    print(f"Saved figure '{name}' -> {', '.join(p.suffix for p in saved)}")
    return saved


def save_table(df, name):
    """Save a results table as CSV (Excel-readable) and as a LaTeX table for the manuscript."""
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    try:
        df.to_latex(out_dir / f"{name}.tex", index=False, float_format="%.2f")
    except Exception:
        pass  # LaTeX export is optional
    print(f"Saved table '{name}' -> results/{name}.csv (+ .tex)")


if __name__ == "__main__":
    # Demo: creates figures/demo_figure.* so you can check that everything works.
    import pandas as pd

    apply_style()
    df = pd.DataFrame({
        "shots": [0, 1, 5, 10, 16] * 2,
        "accuracy": [20, 35, 50, 58, 63, 30, 48, 62, 70, 74],
        "model": ["Model A"] * 5 + ["Model B"] * 5,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for model, g in df.groupby("model"):
        ax.plot(g["shots"], g["accuracy"], marker="o", label=model)
    ax.set_xlabel("Field images per class (shots)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    save_figure(fig, "demo_figure", data=df)
