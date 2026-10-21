"""
Sample images from the four datasets, for the Methods section (one figure, editable formats).

Rows: PlantVillage (lab) | PlantDoc (web field) | Plant Pathology 2020 (orchard) | JIC wheat (new crop).
The PlantVillage and PlantDoc rows show the SAME five diseases, so readers see the lab-to-field shift.

Run from the project folder (plant-disease-fm):
    python src/make_sample_figure.py              # picks images with a fixed seed
    python src/make_sample_figure.py --seed 3     # different random picks
    python src/make_sample_figure.py --from-list  # use the paths in results/sample_images.csv
                                                  # (edit that file to swap any image you do not like)

PlantDoc images come from the web: the script skips file names that look like stock photos and
images listed in the label audit, but please LOOK at the figure and swap any image that shows a
watermark or logo (edit results/sample_images.csv, then run again with --from-list).

Output: figures/final/Fig_dataset_examples.(svg/pdf/eps/png/tiff) and results/sample_images.csv
Running time: a few seconds.
"""
import argparse
import csv
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import figstyle  # noqa: E402
import fm  # noqa: E402

LIST = fm.ROOT / "results" / "sample_images.csv"
SHARED = [("tomato", "early_blight"), ("tomato", "late_blight"), ("corn", "common_rust"),
          ("potato", "late_blight"), ("apple", "scab")]
STOCK = ["stock", "shutterstock", "alamy", "dreamstime", "istock", "depositphotos", "123rf",
         "getty", "fotolia", "bigstock", "watermark", "sciencephoto", "science-photo", "agefotostock", "mediastore"]
ROWS = [("PlantVillage\n(laboratory)", "plantvillage_test_sub"), ("PlantDoc\n(web field photos)", "plantdoc_test"),
        ("Plant Pathology 2020\n(apple orchard)", "pp_test"), ("JIC wheat\n(new crop)", "wheat_test")]
N_COLS = 5


def short_names():
    try:
        return fm.class_list()[1]
    except Exception:
        return {}


def nice(label, names):
    if label in names:
        return names[label]
    return label.replace("___", " ").replace("_", " ").strip().capitalize()


def audited():
    p = fm.ROOT / "results" / "analysis" / "plantdoc_label_audit.csv"
    if not p.exists():
        return set()
    with open(p, encoding="utf-8") as f:
        return {r["path"] for r in csv.DictReader(f)}


def find_label(labels, crop, disease):
    for lab in sorted(labels):
        low = lab.lower()
        if crop in low and disease in low:
            return lab
    return None


def pick(seed):
    rng = random.Random(seed)
    bad = audited()
    chosen = []
    for row_name, split in ROWS:
        rows = fm.read_split(split)
        if not rows:
            print(f"  {split}: split not found - row skipped")
            continue
        by = {}
        for r in rows:
            name = r["path"].lower()
            if split == "plantdoc_test" and (r["path"] in bad or any(s in name for s in STOCK)):
                continue
            by.setdefault(r["label"], []).append(r["path"])
        labels = list(by)
        if split in ("plantvillage_test_sub", "plantdoc_test"):
            wanted = [find_label(labels, c, d) for c, d in SHARED]
        else:
            wanted = sorted(labels)
            while len(wanted) < N_COLS:                 # Plant Pathology has 3 classes: repeat scab and rust
                wanted += [l for l in sorted(labels) if "healthy" not in l.lower()][: N_COLS - len(wanted)]
        used = set()
        for lab in wanted[:N_COLS]:
            if lab is None or lab not in by:
                chosen.append({"row": row_name, "split": split, "label": "", "path": ""})
                continue
            pool = [p for p in by[lab] if p not in used]
            p = rng.choice(pool)
            used.add(p)
            chosen.append({"row": row_name, "split": split, "label": lab, "path": p})
    LIST.parent.mkdir(parents=True, exist_ok=True)
    with open(LIST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row", "split", "label", "path"])
        w.writeheader()
        w.writerows(chosen)
    return chosen


def thumb(path, size=320):
    with Image.open(fm.ROOT / path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(ImageOps.fit(im, (size, size), Image.BICUBIC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-list", action="store_true")
    args = ap.parse_args()
    if args.from_list and LIST.exists():
        with open(LIST, encoding="utf-8") as f:
            chosen = list(csv.DictReader(f))
    else:
        chosen = pick(args.seed)
    names = short_names()
    rows = []
    for r in chosen:
        if r["row"] not in rows:
            rows.append(r["row"])

    figstyle.apply_style()
    fig, axes = plt.subplots(len(rows), N_COLS, figsize=(7.2, 1.62 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, row in enumerate(rows):
        items = [r for r in chosen if r["row"] == row][:N_COLS]
        for j in range(N_COLS):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j < len(items) and items[j]["path"]:
                try:
                    ax.imshow(thumb(items[j]["path"]))
                except Exception as e:
                    print("  could not open", items[j]["path"], e)
                ax.set_title(nice(items[j]["label"], names), fontsize=6.5, pad=2)
        axes[i, 0].set_ylabel(row, fontsize=7, rotation=90, labelpad=6)
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    figstyle.FIG_DIR = fm.ROOT / "figures" / "final"
    figstyle.save_figure(fig, "Fig_dataset_examples")
    print("\nImages used (edit results/sample_images.csv and run with --from-list to swap any):")
    for r in chosen:
        print(f"  {r['split']:<22} {nice(r['label'], names):<28} {r['path']}")
    print("\nSaved: figures/final/Fig_dataset_examples.png (+ svg, pdf, eps, tiff)")


if __name__ == "__main__":
    main()
