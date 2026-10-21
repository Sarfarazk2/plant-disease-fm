"""
Step 10a - Prepare the John Innes Centre wheat disease images (third, untouched field dataset).
Running time: about 3-8 minutes (images are resized once).

Wheat does not occur in PlantVillage, so this dataset is used for:
  * a NEW-CROP few-shot test (confirms the fusion method v2 on data never used to design it)
  * the UNKNOWN-DISEASE (open-set) test (Step 11)

Run from the project folder (plant-disease-fm):
    python src/prepare_wheat.py

Expects the 999 JPGs extracted (any sub-folder depth) inside data/wheat_jic/.
The label is the letters at the start of each file name, e.g. "BrownRust91.jpg" -> brown rust.

Output:
    data/wheat_512/<label>/<name>.jpg
    data/splits/wheat_test.csv, wheat_pool.csv, wheat_fewshot_k{K}_seed{S}.csv
    results/wheat_split_report.txt
"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from prepare_splits import NEAR_DUP_MAX_BITS, dhash

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "wheat_jic"
OUT = ROOT / "data" / "wheat_512"
SPLITS = ROOT / "data" / "splits"
MAX_SIDE = 512
POOL_PER_CLASS = 30
SHOTS, SEEDS = [1, 5, 10, 16], [0, 1, 2, 3, 4]
IMG_EXT = {".jpg", ".jpeg", ".png"}

# file-name prefix (lower case, letters only) -> (label, short name)
PREFIXES = {
    "brownrust": ("Wheat___Brown_rust", "Wheat brown rust"),
    "yellowrust": ("Wheat___Yellow_rust", "Wheat yellow rust"),
    "septoria": ("Wheat___Septoria_leaf_blotch", "Wheat septoria"),
    "mildew": ("Wheat___Powdery_mildew", "Wheat powdery mildew"),
    "powderymildew": ("Wheat___Powdery_mildew", "Wheat powdery mildew"),
    "healthy": ("Wheat___healthy", "Wheat healthy"),
}
WHEAT_CLASSES = sorted({v[0] for v in PREFIXES.values()})


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "short_name"])
        w.writeheader()
        w.writerows(rows)


def main():
    files = sorted(p for p in SRC.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"No images found inside {SRC}")
    prefix_of = {p: re.match(r"[A-Za-z]+", p.stem).group(0).lower() if re.match(r"[A-Za-z]+", p.stem) else ""
                 for p in files}
    counts = Counter(prefix_of.values())
    print("File-name prefixes found:", dict(counts))
    unknown = [k for k in counts if k not in PREFIXES]
    if unknown:
        raise SystemExit(f"Unknown prefixes {unknown} - add them to PREFIXES.")

    rows = []
    for p in tqdm(files, desc="Resizing to 512 px"):
        label, short = PREFIXES[prefix_of[p]]
        dst = OUT / label / f"{p.stem}.jpg"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")   # respect phone orientation
                im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                im.save(dst, quality=95)
        rows.append({"path": dst.relative_to(ROOT).as_posix(), "label": label, "short_name": short})

    rng = np.random.default_rng(0)
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r)
    pool, test = [], []
    for lab in sorted(by_class):
        rs = sorted(by_class[lab], key=lambda r: r["path"])
        idx = rng.permutation(len(rs))
        n_pool = min(POOL_PER_CLASS, len(rs) // 2)
        pool += [rs[i] for i in idx[:n_pool]]
        test += [rs[i] for i in idx[n_pool:]]

    ph_test = np.array([dhash(ROOT / r["path"]) for r in tqdm(test, desc="hash test")])
    clean_pool, removed = [], 0
    for r in tqdm(pool, desc="hash pool"):
        if (ph_test != dhash(ROOT / r["path"])).sum(axis=1).min() <= NEAR_DUP_MAX_BITS:
            removed += 1
        else:
            clean_pool.append(r)

    write_csv(SPLITS / "wheat_test.csv", test)
    write_csv(SPLITS / "wheat_pool.csv", clean_pool)
    pool_by = defaultdict(list)
    for r in clean_pool:
        pool_by[r["label"]].append(r)
    for k in SHOTS:
        for s in SEEDS:
            r2 = np.random.default_rng(s)
            sel = []
            for lab in sorted(pool_by):
                p = pool_by[lab]
                sel += [p[i] for i in r2.choice(len(p), size=min(k, len(p)), replace=False)]
            write_csv(SPLITS / f"wheat_fewshot_k{k}_seed{s}.csv", sel)

    report = [f"JIC wheat: {len(rows)} images in {len(by_class)} classes"]
    for lab in sorted(by_class):
        nt = sum(r["label"] == lab for r in test)
        report.append(f"  {lab:<30} total {len(by_class[lab]):>4} | pool {len(pool_by[lab]):>3} | test {nt:>4}")
    report.append(f"Pool images removed as near-duplicates of test images: {removed}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "wheat_split_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
