"""
Step 3a - Download PlantVillage (color images, official leaf-grouped split).

Run from the project folder (plant-disease-fm):
    python src/download_plantvillage.py

What it does:
  1. Downloads data.zip (about 2.2 GB) plus the official split lists
     (splits/color_train.txt, splits/color_test.txt) from huggingface.co/datasets/mohanty/PlantVillage.
     The download resumes automatically if it is interrupted - just run the command again.
  2. Extracts ONLY the color images listed in those split files.
     The official split keeps all photos of the same physical leaf in the same split,
     so there is no train/test leakage.

Output:
    data/plantvillage/train/<class_name>/<original file name>
    data/plantvillage/test/<class_name>/<original file name>
    data/plantvillage/metadata.csv   (split, path, label, crop, disease, source_path)

Needs about 4 GB of free disk space. After it prints "Done", you may delete the
download cache folder printed at the end to recover ~2.2 GB.
"""
import csv
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "plantvillage"
REPO = "mohanty/PlantVillage"


def fetch(filename):
    return Path(hf_hub_download(repo_id=REPO, filename=filename, repo_type="dataset"))


def read_split(path):
    with open(path, encoding="utf-8") as f:
        lines = [line.strip().replace("\\", "/") for line in f]
    return [ln for ln in lines if ln.startswith("raw/")]


def normalise(name):
    """Return the part of a zip path starting at 'raw/' (handles any top-level folder)."""
    name = name.replace("\\", "/")
    idx = name.find("raw/")
    return name[idx:] if idx >= 0 else None


def main():
    print("Downloading split lists ...")
    splits = {
        "train": read_split(fetch("splits/color_train.txt")),
        "test": read_split(fetch("splits/color_test.txt")),
    }
    for s, items in splits.items():
        print(f"  {s}: {len(items)} images listed")

    print("Downloading data.zip (about 2.2 GB). This can take a long time on a slow connection;")
    print("if it stops, run the same command again and it will continue.")
    zip_path = fetch("data.zip")
    print(f"  saved at: {zip_path}")

    rows, missing = [], 0
    with zipfile.ZipFile(zip_path) as zf:
        index = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            key = normalise(info.filename)
            if key and key.startswith("raw/color/"):
                index[key] = info

        for split, items in splits.items():
            for rel in tqdm(items, desc=f"Extracting {split}"):
                info = index.get(rel)
                if info is None:
                    missing += 1
                    continue
                parts = rel.split("/")          # raw / color / <class> / <file>
                label, fname = parts[2], parts[-1]
                out_path = OUT / split / label / fname
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if not out_path.exists():
                    out_path.write_bytes(zf.read(info))
                crop, _, disease = label.partition("___")
                rows.append({
                    "split": split,
                    "path": out_path.relative_to(ROOT).as_posix(),
                    "label": label,
                    "crop": crop,
                    "disease": disease or "unknown",
                    "source_path": rel,
                })

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Images saved   : {len(rows)}")
    print(f"Classes        : {len({r['label'] for r in rows})}")
    print(f"Not found in zip (should be 0): {missing}")
    print(f"Metadata       : {OUT / 'metadata.csv'}")
    print(f"Download cache (can be deleted to free space): {zip_path.parent}")


if __name__ == "__main__":
    main()
