"""
Step 4 - Class matching, duplicate removal and fixed few-shot splits.
Running time: about 2-5 minutes.

Run from the project folder (plant-disease-fm):
    python src/prepare_splits.py

What it does
  1. Maps the 28 PlantDoc (field) classes to their PlantVillage (lab) classes.
     The PlantVillage name is used as the single official label everywhere.
  2. Finds PlantDoc TRAIN images that are exact or near-duplicates of TEST images
     (PlantDoc was collected from the web, so copies exist) and removes them from
     the training pool. The test set itself is NOT changed, so our results stay
     comparable with published PlantDoc numbers.
  3. Creates fixed few-shot training sets (K = 1, 5, 10, 16 images per class,
     5 random repeats each). Every experiment uses exactly these files, which makes
     all comparisons fair and fully reproducible.

Output (all in data/splits/):
    class_mapping.csv              PlantDoc name -> PlantVillage label, short name
    plantvillage_train.csv / _test.csv
    plantdoc_train_clean.csv       PlantDoc train pool after duplicate removal
    plantdoc_test.csv              PlantDoc test set (unchanged)
    fewshot_k{K}_seed{S}.csv       few-shot sets drawn from plantdoc_train_clean
    duplicates_removed.csv         every removed image and the test image it copies
Also: results/split_report.txt
"""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS = DATA / "splits"
RESULTS = ROOT / "results"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

SHOTS = [1, 5, 10, 16]
SEEDS = [0, 1, 2, 3, 4]
NEAR_DUP_MAX_BITS = 5   # perceptual-hash distance (out of 64 bits) counted as "near-duplicate"

# PlantDoc folder name -> (PlantVillage label, short name for figures)
MAPPING = {
    "Apple Scab Leaf": ("Apple___Apple_scab", "Apple scab"),
    "Apple leaf": ("Apple___healthy", "Apple healthy"),
    "Apple rust leaf": ("Apple___Cedar_apple_rust", "Apple rust"),
    "Bell_pepper leaf": ("Pepper,_bell___healthy", "Pepper healthy"),
    "Bell_pepper leaf spot": ("Pepper,_bell___Bacterial_spot", "Pepper bacterial spot"),
    "Blueberry leaf": ("Blueberry___healthy", "Blueberry healthy"),
    "Cherry leaf": ("Cherry_(including_sour)___healthy", "Cherry healthy"),
    "Corn Gray leaf spot": ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn gray leaf spot"),
    "Corn leaf blight": ("Corn_(maize)___Northern_Leaf_Blight", "Corn northern leaf blight"),
    "Corn rust leaf": ("Corn_(maize)___Common_rust_", "Corn common rust"),
    "Peach leaf": ("Peach___healthy", "Peach healthy"),
    "Potato leaf early blight": ("Potato___Early_blight", "Potato early blight"),
    "Potato leaf late blight": ("Potato___Late_blight", "Potato late blight"),
    "Raspberry leaf": ("Raspberry___healthy", "Raspberry healthy"),
    "Soyabean leaf": ("Soybean___healthy", "Soybean healthy"),
    "Squash Powdery mildew leaf": ("Squash___Powdery_mildew", "Squash powdery mildew"),
    "Strawberry leaf": ("Strawberry___healthy", "Strawberry healthy"),
    "Tomato Early blight leaf": ("Tomato___Early_blight", "Tomato early blight"),
    "Tomato Septoria leaf spot": ("Tomato___Septoria_leaf_spot", "Tomato septoria leaf spot"),
    "Tomato leaf": ("Tomato___healthy", "Tomato healthy"),
    "Tomato leaf bacterial spot": ("Tomato___Bacterial_spot", "Tomato bacterial spot"),
    "Tomato leaf late blight": ("Tomato___Late_blight", "Tomato late blight"),
    "Tomato leaf mosaic virus": ("Tomato___Tomato_mosaic_virus", "Tomato mosaic virus"),
    "Tomato leaf yellow virus": ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato yellow leaf curl virus"),
    "Tomato mold leaf": ("Tomato___Leaf_Mold", "Tomato leaf mold"),
    "Tomato two spotted spider mites leaf": ("Tomato___Spider_mites Two-spotted_spider_mite", "Tomato spider mites"),
    "grape leaf": ("Grape___healthy", "Grape healthy"),
    "grape leaf black rot": ("Grape___Black_rot", "Grape black rot"),
}


def find_split_root(dataset_dir):
    if (dataset_dir / "train").is_dir():
        return dataset_dir
    for child in dataset_dir.iterdir():
        if child.is_dir() and (child / "train").is_dir():
            return child
    raise FileNotFoundError(f"No 'train' folder found inside {dataset_dir}")


def list_split(split_dir):
    """Return {class_folder_name: [image paths]}."""
    out = {}
    for d in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        out[d.name.strip()] = sorted(p for p in d.rglob("*")
                                     if p.is_file() and p.suffix.lower() in IMG_EXT)
    return out


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def dhash(path, size=8):
    """64-bit difference hash: robust to resizing and re-compression."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    report = []

    def log(msg=""):
        print(msg)
        report.append(msg)

    # ---------- 1. class mapping ----------
    pd_root = find_split_root(DATA / "plantdoc")
    pd_train = list_split(pd_root / "train")
    pd_test = list_split(pd_root / "test")
    pv_train = list_split(DATA / "plantvillage" / "train")
    pv_test = list_split(DATA / "plantvillage" / "test")

    unknown = sorted((set(pd_train) | set(pd_test)) - set(MAPPING))
    if unknown:
        raise SystemExit(f"These PlantDoc folders are not in MAPPING (add them): {unknown}")
    missing_pv = sorted({v[0] for v in MAPPING.values()} - set(pv_train))
    if missing_pv:
        raise SystemExit(f"These PlantVillage classes were not found (check the download): {missing_pv}")

    shared = sorted({MAPPING[k][0] for k in pd_train})
    short = {v[0]: v[1] for v in MAPPING.values()}
    write_csv(SPLITS / "class_mapping.csv",
              [{"plantdoc_name": k, "label": v[0], "short_name": v[1],
                "plantdoc_train": len(pd_train.get(k, [])), "plantdoc_test": len(pd_test.get(k, [])),
                "plantvillage_train": len(pv_train[v[0]]), "plantvillage_test": len(pv_test[v[0]])}
               for k, v in sorted(MAPPING.items(), key=lambda kv: kv[1][0])],
              ["plantdoc_name", "label", "short_name", "plantdoc_train", "plantdoc_test",
               "plantvillage_train", "plantvillage_test"])
    log(f"[1] Class mapping: {len(MAPPING)} PlantDoc classes -> PlantVillage labels")
    log(f"    Shared classes used in experiments: {len(shared)}")
    log(f"    PlantVillage-only classes (kept aside for later use): {len(set(pv_train) - set(shared))}")

    # PlantVillage csv files (all 38 classes, with a 'shared' flag)
    for name, split in [("plantvillage_train", pv_train), ("plantvillage_test", pv_test)]:
        rows = [{"path": rel(p), "label": lab, "short_name": short.get(lab, lab),
                 "shared": int(lab in shared)} for lab, ps in split.items() for p in ps]
        write_csv(SPLITS / f"{name}.csv", rows, ["path", "label", "short_name", "shared"])

    # ---------- 2. duplicates between PlantDoc train and test ----------
    train_items = [(p, MAPPING[k][0]) for k, ps in pd_train.items() for p in ps]
    test_items = [(p, MAPPING[k][0]) for k, ps in pd_test.items() for p in ps]

    log("\n[2] Hashing PlantDoc images to find duplicates ...")
    def hashes(items, desc):
        exact, perc = [], []
        for p, _ in tqdm(items, desc=desc):
            exact.append(md5(p))
            try:
                perc.append(dhash(p))
            except Exception:
                perc.append(np.zeros(64, dtype=bool))
        return exact, np.array(perc)

    tr_md5, tr_ph = hashes(train_items, "train")
    te_md5, te_ph = hashes(test_items, "test")

    te_md5_index = {h: i for i, h in enumerate(te_md5)}
    # Hamming distance between every train and test image (2.3k x 0.24k is small)
    dist = (tr_ph[:, None, :] != te_ph[None, :, :]).sum(axis=2)

    removed, keep = [], []
    for i, (p, lab) in enumerate(train_items):
        j_exact = te_md5_index.get(tr_md5[i])
        j_near = int(dist[i].argmin()) if len(te_ph) else None
        if j_exact is not None:
            kind, j, d = "exact", j_exact, 0
        elif j_near is not None and dist[i, j_near] <= NEAR_DUP_MAX_BITS:
            kind, j, d = "near", j_near, int(dist[i, j_near])
        else:
            keep.append((p, lab))
            continue
        removed.append({"train_path": rel(p), "train_label": lab,
                        "test_path": rel(test_items[j][0]), "test_label": test_items[j][1],
                        "type": kind, "hash_distance": d,
                        "same_label": int(lab == test_items[j][1])})

    write_csv(SPLITS / "duplicates_removed.csv", removed,
              ["train_path", "train_label", "test_path", "test_label", "type",
               "hash_distance", "same_label"])
    n_exact = sum(r["type"] == "exact" for r in removed)
    n_label_conflict = sum(not r["same_label"] for r in removed)
    log(f"    Train images checked: {len(train_items)}, test images: {len(test_items)}")
    log(f"    Removed from train pool: {len(removed)} "
        f"({n_exact} exact copies, {len(removed) - n_exact} near-duplicates)")
    log(f"    Of these, copies whose label differs from the test copy: {n_label_conflict}")

    # within-test duplicates (reported only, test set unchanged)
    test_dup_pairs = 0
    if len(te_ph) > 1:
        dt = (te_ph[:, None, :] != te_ph[None, :, :]).sum(axis=2)
        test_dup_pairs = int(((dt <= NEAR_DUP_MAX_BITS).sum() - len(te_ph)) // 2)
    log(f"    Duplicate pairs inside the test set (reported, not changed): {test_dup_pairs}")

    write_csv(SPLITS / "plantdoc_train_clean.csv",
              [{"path": rel(p), "label": lab, "short_name": short[lab]} for p, lab in keep],
              ["path", "label", "short_name"])
    write_csv(SPLITS / "plantdoc_test.csv",
              [{"path": rel(p), "label": lab, "short_name": short[lab]} for p, lab in test_items],
              ["path", "label", "short_name"])

    # ---------- 3. few-shot splits ----------
    by_class = defaultdict(list)
    for p, lab in keep:
        by_class[lab].append(rel(p))
    log("\n[3] Few-shot training sets (from the cleaned PlantDoc train pool)")
    short_classes = {}
    for k in SHOTS:
        for s in SEEDS:
            rng = np.random.default_rng(s)
            rows = []
            for lab in shared:
                pool = sorted(by_class.get(lab, []))
                take = min(k, len(pool))
                if take < k:
                    short_classes.setdefault(k, set()).add(f"{short[lab]} ({len(pool)})")
                for idx in rng.choice(len(pool), size=take, replace=False) if take else []:
                    rows.append({"path": pool[idx], "label": lab, "short_name": short[lab]})
            write_csv(SPLITS / f"fewshot_k{k}_seed{s}.csv", rows, ["path", "label", "short_name"])
        log(f"    K={k:>2}: {len(SEEDS)} repeats x {len(shared)} classes"
            + (f" | classes with fewer than {k} images (all used): {', '.join(sorted(short_classes[k]))}"
               if k in short_classes else ""))

    # ---------- summary ----------
    test_counts = Counter(lab for _, lab in test_items)
    absent = [short[c] for c in shared if test_counts[c] == 0]
    log("\n[Summary]")
    log(f"    PlantVillage train/test: {sum(map(len, pv_train.values()))} / {sum(map(len, pv_test.values()))}")
    log(f"    PlantDoc train pool after cleaning: {len(keep)} (was {len(train_items)})")
    log(f"    PlantDoc test: {len(test_items)} images, {len(test_counts)} classes")
    if absent:
        log(f"    Classes with NO test images (excluded from test metrics): {', '.join(absent)}")
    log(f"\nAll files saved in {rel(SPLITS)}")

    (RESULTS / "split_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("Report saved to results/split_report.txt")


if __name__ == "__main__":
    main()
