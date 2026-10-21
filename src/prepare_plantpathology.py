"""
Step 8a - Prepare Plant Pathology 2020 (FGVC7) as a second, NON-WEB field test set.
Running time: about 5-10 minutes (images are resized once).

Images were taken in apple orchards (Cornell), not collected from the web, so they help
test whether web-trained models benefit from having seen PlantDoc-like web images.

Run from the project folder (plant-disease-fm):
    python src/prepare_plantpathology.py

Expects the Kaggle ZIP extracted somewhere inside data/plantpathology/
(train.csv + images/ folder).

What it does
  1. Keeps only single-label images of the 3 classes shared with our benchmark:
       scab -> Apple___Apple_scab, rust -> Apple___Cedar_apple_rust, healthy -> Apple___healthy
     ("multiple_diseases" images are excluded).
  2. Resizes each image ONCE so its longer side is 512 px (all models use 224 px input, so this
     does not change results but makes feature extraction ~5x faster).
  3. Splits per class: a fixed support pool of 50 images (few-shot examples are drawn from it)
     and the rest as the test set. Near-duplicate pool images of test images are removed.
  4. Few-shot files: K = 1, 5, 10, 16 per class, 5 seeds.

Output:
    data/plantpathology_512/<label>/<image_id>.jpg
    data/splits/pp_test.csv, pp_pool.csv, pp_fewshot_k{K}_seed{S}.csv
    results/pp_split_report.txt
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from prepare_splits import NEAR_DUP_MAX_BITS, dhash

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "plantpathology"
OUT = ROOT / "data" / "plantpathology_512"
SPLITS = ROOT / "data" / "splits"
MAX_SIDE = 512
POOL_PER_CLASS = 50
SHOTS, SEEDS = [1, 5, 10, 16], [0, 1, 2, 3, 4]
LABELS = {"scab": ("Apple___Apple_scab", "Apple scab"),
          "rust": ("Apple___Cedar_apple_rust", "Apple rust"),
          "healthy": ("Apple___healthy", "Apple healthy")}


def find_inputs():
    csvs = list(SRC.rglob("train.csv"))
    if not csvs:
        raise SystemExit(f"train.csv not found inside {SRC}. Extract the Kaggle ZIP there.")
    train_csv = csvs[0]
    img_dir = train_csv.parent / "images"
    if not img_dir.is_dir():
        found = [p.parent for p in SRC.rglob("Train_0.jpg")]
        if not found:
            raise SystemExit("images folder (with Train_0.jpg ...) not found.")
        img_dir = found[0]
    return train_csv, img_dir


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "short_name"])
        w.writeheader()
        w.writerows(rows)


def main():
    train_csv, img_dir = find_inputs()
    with open(train_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    items, skipped = [], 0
    for r in rows:
        flags = {k: int(float(r[k])) for k in ["healthy", "multiple_diseases", "rust", "scab"]}
        on = [k for k, v in flags.items() if v == 1]
        if len(on) != 1 or on[0] == "multiple_diseases":
            skipped += 1
            continue
        items.append((r["image_id"], on[0]))
    print(f"Single-label images kept: {len(items)} (excluded {skipped})")

    # resize once
    out_rows = []
    for image_id, cls in tqdm(items, desc="Resizing to 512 px"):
        label, short = LABELS[cls]
        dst = OUT / label / f"{image_id}.jpg"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(img_dir / f"{image_id}.jpg") as im:
                im = im.convert("RGB")
                im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                im.save(dst, quality=95)
        out_rows.append({"path": dst.relative_to(ROOT).as_posix(), "label": label, "short_name": short})

    # per-class pool / test split
    rng = np.random.default_rng(0)
    by_class = defaultdict(list)
    for r in out_rows:
        by_class[r["label"]].append(r)
    pool, test = [], []
    for lab in sorted(by_class):
        rs = sorted(by_class[lab], key=lambda r: r["path"])
        idx = rng.permutation(len(rs))
        pool += [rs[i] for i in idx[:POOL_PER_CLASS]]
        test += [rs[i] for i in idx[POOL_PER_CLASS:]]

    # remove pool images that are near-duplicates of test images
    print("Checking for near-duplicates between pool and test ...")
    ph_test = np.array([dhash(ROOT / r["path"]) for r in tqdm(test, desc="hash test")])
    clean_pool, removed = [], 0
    for r in tqdm(pool, desc="hash pool"):
        d = (ph_test != dhash(ROOT / r["path"])).sum(axis=1)
        if d.min() <= NEAR_DUP_MAX_BITS:
            removed += 1
        else:
            clean_pool.append(r)

    write_csv(SPLITS / "pp_test.csv", test)
    write_csv(SPLITS / "pp_pool.csv", clean_pool)
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
            write_csv(SPLITS / f"pp_fewshot_k{k}_seed{s}.csv", sel)

    report = [f"Plant Pathology 2020: {len(out_rows)} single-label images in 3 classes"]
    for lab in sorted(by_class):
        nt = sum(r["label"] == lab for r in test)
        report.append(f"  {lab:<28} total {len(by_class[lab]):>4} | pool {len(pool_by[lab]):>3} | test {nt:>4}")
    report.append(f"Pool images removed as near-duplicates of test images: {removed}")
    report.append(f"Test set: {len(test)} images; few-shot files: K={SHOTS} x {len(SEEDS)} seeds")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "pp_split_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
