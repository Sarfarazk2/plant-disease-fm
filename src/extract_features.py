"""
Step 6 - Extract image features for every model and every data split (LONG RUN).

Run from the project folder (plant-disease-fm):
    python src/extract_features.py                       # all models, field + lab subsets (~1.5-2 h)
    python src/extract_features.py --workers 2           # same, with 2 image-loading helpers (faster)
    python src/extract_features.py --models clip_b16     # one model only
    python src/extract_features.py --splits plantdoc_test plantdoc_train_clean   # small splits only
    python src/extract_features.py --splits pp_test pp_pool --workers 2   # Plant Pathology (Step 8)

Safe to stop at any time with Ctrl+C (or if the battery runs out):
work is saved every 2,048 images, and running the same command again continues
where it stopped. Finished model/split combinations are skipped instantly.

Order: small field splits first, then the large PlantVillage (lab) splits, so the
most important features are ready earliest.

Output: features/<model>/<split>.npz   and   results/feature_extraction_log.csv
"""
import argparse
import csv
import time

import fm

# Default: field splits + the fixed balanced lab subsets (Step 6a).
# Full lab splits ("plantvillage_test", "plantvillage_train") only for the CLIP robustness check.
SPLITS = ["plantdoc_test", "plantdoc_train_clean", "plantvillage_test_sub", "plantvillage_train_sub"]
ORDER = list(fm.FOUNDATION)   # CNN baselines: python src/extract_features.py --models resnet50 effnet_b0 mobilenet_v3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ORDER, choices=list(fm.MODELS))
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--workers", type=int, default=0,
                    help="image-loading worker processes (0 = safest on Windows)")
    args = ap.parse_args()

    print(f"Device: {fm.DEVICE}")
    log_path = fm.ROOT / "results" / "feature_extraction_log.csv"
    log_path.parent.mkdir(exist_ok=True)
    new_file = not log_path.exists()

    for m in args.models:
        for s in args.splits:
            t0 = time.time()
            feats, labels, _ = fm.encode_images(m, s, num_workers=args.workers)
            secs = time.time() - t0
            print(f"  done: {m:<11} | {s:<22} | {feats.shape[0]:>6} images | "
                  f"dim {feats.shape[1]:>4} | {secs / 60:.1f} min")
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["model", "split", "n_images", "dim", "seconds"])
                    new_file = False
                w.writerow([m, s, feats.shape[0], feats.shape[1], f"{secs:.1f}"])
        fm.free_model(m)
    print("\nAll requested features are ready in the 'features' folder.")


if __name__ == "__main__":
    main()
