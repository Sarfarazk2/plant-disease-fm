"""
Step 3b - Check the downloaded datasets: count images per class and find broken files.

Run from the project folder (plant-disease-fm):
    python src/inspect_datasets.py

It looks for:
    data/plantvillage/train, data/plantvillage/test
    data/plantdoc/train,     data/plantdoc/test
(PlantDoc may sit one level deeper, e.g. data/plantdoc/PlantDoc-Dataset-master/train;
 the script finds it automatically.)

Output:
    results/dataset_summary.csv   (dataset, split, class, n_images)
    results/bad_images.txt        (files that cannot be opened, if any)
    A short summary printed on screen.
"""
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Datasets smaller than this are fully checked image-by-image for corruption.
FULL_CHECK_LIMIT = 15000


def find_split_root(dataset_dir: Path):
    """Return the folder that directly contains 'train' (searches up to 2 levels)."""
    if (dataset_dir / "train").is_dir():
        return dataset_dir
    for child in dataset_dir.iterdir() if dataset_dir.is_dir() else []:
        if child.is_dir() and (child / "train").is_dir():
            return child
    return None


def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]


def check_image(path: Path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main():
    RESULTS.mkdir(exist_ok=True)
    summary_rows, bad_files = [], []

    for name in ["plantvillage", "plantdoc"]:
        base = find_split_root(DATA / name)
        if base is None:
            print(f"\n[{name}] NOT FOUND - expected a 'train' folder inside data/{name}")
            continue

        print(f"\n[{name}] found at {base.relative_to(ROOT)}")
        for split in ["train", "test"]:
            split_dir = base / split
            if not split_dir.is_dir():
                print(f"  {split}: missing")
                continue

            counts = defaultdict(int)
            all_imgs = []
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                imgs = list_images(class_dir)
                counts[class_dir.name] = len(imgs)
                all_imgs.extend(imgs)

            do_full_check = len(all_imgs) <= FULL_CHECK_LIMIT
            if do_full_check:
                for p in all_imgs:
                    if not check_image(p):
                        bad_files.append(p.relative_to(ROOT).as_posix())

            for cls, n in counts.items():
                summary_rows.append({"dataset": name, "split": split, "class": cls, "n_images": n})

            n_total = sum(counts.values())
            smallest = min(counts.items(), key=lambda kv: kv[1]) if counts else ("-", 0)
            print(f"  {split}: {n_total} images, {len(counts)} classes "
                  f"(smallest class: {smallest[0]} = {smallest[1]})"
                  + ("" if do_full_check else "  [corruption check skipped: large split]"))

        # Print class names once per dataset (needed to build the class mapping)
        classes = sorted({r["class"] for r in summary_rows if r["dataset"] == name})
        print(f"  class names ({len(classes)}):")
        for c in classes:
            print("    -", c)

    with open(RESULTS / "dataset_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "split", "class", "n_images"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(RESULTS / "bad_images.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(bad_files))

    print(f"\nBroken image files found: {len(bad_files)}"
          + (" (listed in results/bad_images.txt)" if bad_files else ""))
    print("Summary saved to results/dataset_summary.csv")


if __name__ == "__main__":
    main()
