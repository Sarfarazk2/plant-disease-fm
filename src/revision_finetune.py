"""
REVISION R1 - Fully fine-tuned CNN baselines with a proper LAB VALIDATION set and several seeds.

Why: in the first version the best epoch was chosen on the lab TEST subset. Here the epoch is
chosen on a separate lab validation set (50 images per class, drawn from the official PlantVillage
training split and disjoint from the 200-per-class training subset), and every model is trained
with 3 seeds. Field accuracy is also recorded after every epoch (NEVER used for selection) to show
how much field accuracy changes while lab accuracy stays flat.

LONG RUN: about 6-7 hours for all 3 CNNs x 3 seeds on a 4 GB laptop GPU. Safe to stop at any time:
re-running the same command continues from the last finished epoch / skips finished runs.

Run from the project folder (plant-disease-fm):
    python src/revision_finetune.py                                  # all 3 CNNs, seeds 0 1 2
    python src/revision_finetune.py --models resnet50 --seeds 0      # one run only

Output: data/splits/plantvillage_val.csv (created once)
        models/rev_<model>_s<seed>_{last,best}.pt
        results/revision/finetune/epochs_<model>_s<seed>.csv   (every epoch: val, lab test, field)
        results/revision/finetune/final_<model>_s<seed>.csv    (selected epoch)
        results/revision/finetune/logits_<model>_s<seed>.npz   (for calibration / statistics)
        results/revision/finetune/summary.csv                  (mean +- SD over seeds)
"""
import argparse
import csv
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

import fm
from finetune_cnn import APPLE, MEAN, STD, LabeledCSV, metrics, predict

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
OUT = fm.ROOT / "results" / "revision" / "finetune"
CKPT = fm.ROOT / "models"
VAL_PER_CLASS = 50
EVAL_SPLITS = [("lab_test", "plantvillage_test_sub"), ("plantdoc", "plantdoc_test"), ("pp", "pp_test")]


def make_val_split():
    """50 images per class from the official PlantVillage TRAIN split, not in the training subset."""
    path = fm.SPLITS / "plantvillage_val.csv"
    if path.exists():
        return fm.read_split("plantvillage_val")
    full = fm.read_split("plantvillage_train")
    used = {r["path"] for r in fm.read_split("plantvillage_train_sub")}
    rng = np.random.default_rng(0)
    classes, _ = fm.class_list()
    rows = []
    for c in classes:
        pool = [r for r in full if r["label"] == c and r["path"] not in used]
        pick = rng.choice(len(pool), size=min(VAL_PER_CLASS, len(pool)), replace=False)
        rows += [pool[i] for i in sorted(pick)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Created data/splits/plantvillage_val.csv ({len(rows)} images, {VAL_PER_CLASS} per class, "
          "disjoint from the training subset)")
    return rows


def accs(model, rows, classes, tf, workers):
    lg, yy = predict(model, rows, classes, tf, 32, workers)
    out = {"28-way": 100 * np.mean(lg.argmax(1) == yy)}
    return out, lg, yy


def apple3(lg, yy, classes):
    ai = [classes.index(c) for c in APPLE]
    return 100 * np.mean(np.array(ai)[lg[:, ai].argmax(1)] == yy)


def run_one(key, seed, args, classes, val_rows):
    final_csv = OUT / f"final_{key}_s{seed}.csv"
    if final_csv.exists():
        print(f"{key} seed {seed}: already finished - skipped")
        return
    import timm
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = timm.create_model(fm.MODELS[key]["name"], pretrained=not args.no_pretrained,
                              num_classes=len(classes)).to(fm.DEVICE)
    train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                          T.RandomVerticalFlip(), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(),
                          T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(LabeledCSV(fm.read_split("plantvillage_train_sub"), classes, train_tf),
                        batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    eval_rows = {name: fm.read_split(split) for name, split in EVAL_SPLITS}

    CKPT.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    last, best_path = CKPT / f"rev_{key}_s{seed}_last.pt", CKPT / f"rev_{key}_s{seed}_best.pt"
    ep_csv = OUT / f"epochs_{key}_s{seed}.csv"
    start, best_val, best_ep, log = 0, -1.0, -1, []
    if last.exists():
        ck = torch.load(last, map_location=fm.DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best_val, best_ep, log = ck["epoch"] + 1, ck["best_val"], ck["best_ep"], ck["log"]
        torch.set_rng_state(ck["rng"])
        print(f"{key} seed {seed}: resuming after epoch {start}")

    for ep in range(start, args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tqdm(loader, desc=f"{key} s{seed} epoch {ep + 1}/{args.epochs}"):
            x, y = x.to(fm.DEVICE), y.to(fm.DEVICE)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        model.eval()
        val = accs(model, val_rows, classes, test_tf, args.workers)[0]["28-way"]
        row = {"model": key, "seed": seed, "epoch": ep + 1, "train_loss": tot / n, "val_accuracy": val}
        if not args.no_field_tracking:          # recorded for analysis only, never used for selection
            for name, rows in eval_rows.items():
                a, lg, yy = accs(model, rows, classes, test_tf, args.workers)
                row[f"{name}_accuracy"] = a["28-way"]
                if name == "pp":
                    row["pp_3way_accuracy"] = apple3(lg, yy, classes)
        log.append(row)
        if val > best_val:
            best_val, best_ep = val, ep + 1
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best_val": best_val, "best_ep": best_ep, "log": log,
                    "rng": torch.get_rng_state()}, last)
        pd.DataFrame(log).round(2).to_csv(ep_csv, index=False)
        extra = "" if args.no_field_tracking else \
            f" | lab test {row['lab_test_accuracy']:.1f} | PlantDoc {row['plantdoc_accuracy']:.1f} | apple {row['pp_accuracy']:.1f}"
        print(f"  epoch {ep + 1}: loss {tot / n:.3f} | VAL {val:.1f}{extra} | {(time.time() - t0) / 60:.1f} min",
              flush=True)

    # ---- evaluate the epoch selected on the VALIDATION set
    model.load_state_dict(torch.load(best_path, map_location=fm.DEVICE, weights_only=True))
    model.eval()
    res, saved = {"model": key, "seed": seed, "selected_epoch": best_ep, "val_accuracy": best_val}, {}
    for name, split in EVAL_SPLITS:
        rows = eval_rows[name]
        order = sorted(r["path"] for r in rows)
        by = {r["path"]: r for r in rows}
        rows = [by[p] for p in order]
        a, lg, yy = accs(model, rows, classes, test_tf, args.workers)
        res[f"{name}_accuracy"] = a["28-way"]
        if name == "pp":
            res["pp_3way_accuracy"] = apple3(lg, yy, classes)
        present = sorted(set(yy.tolist()))
        res[f"{name}_macro_f1"] = metrics(yy, lg.argmax(1), present)["macro_f1"]
        saved.update({f"logits_{split}": lg, f"y_{split}": yy, f"paths_{split}": np.array(order, dtype=object)})
    np.savez(OUT / f"logits_{key}_s{seed}.npz", **saved)
    pd.DataFrame([res]).round(2).to_csv(final_csv, index=False)
    print(f"{key} seed {seed}: selected epoch {best_ep} (val {best_val:.1f}%) -> lab test "
          f"{res['lab_test_accuracy']:.1f} | PlantDoc {res['plantdoc_accuracy']:.1f} | Plant Pathology "
          f"{res['pp_accuracy']:.1f} (3-way {res['pp_3way_accuracy']:.1f})")
    del model
    if fm.DEVICE == "cuda":
        torch.cuda.empty_cache()


def summarise():
    files = sorted(OUT.glob("final_*_s*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cols = ["selected_epoch", "val_accuracy", "lab_test_accuracy", "plantdoc_accuracy", "pp_accuracy",
            "pp_3way_accuracy", "plantdoc_macro_f1", "pp_macro_f1"]
    agg = df.groupby("model")[cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.round(2).reset_index()
    agg.to_csv(OUT / "summary.csv", index=False)
    print("\nFine-tuned CNNs, epoch selected on the lab VALIDATION set (mean +- SD over seeds)")
    for _, r in agg.iterrows():
        n = int(r["lab_test_accuracy_count"])
        f = lambda c: f"{r[c + '_mean']:5.1f} +- {0 if np.isnan(r[c + '_std']) else r[c + '_std']:.1f}"
        print(f"  {fm.MODELS[r['model']]['label']:<28} seeds={n} | lab {f('lab_test_accuracy')} | "
              f"PlantDoc {f('plantdoc_accuracy')} | PP {f('pp_accuracy')} | PP 3-way {f('pp_3way_accuracy')}")
    print("Saved: results/revision/finetune/summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet50", "effnet_b0", "mobilenet_v3"], choices=fm.CNNS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-field-tracking", action="store_true", help="skip per-epoch field evaluation (faster)")
    ap.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)   # for offline testing only
    args = ap.parse_args()
    classes, _ = fm.class_list()
    val_rows = make_val_split()
    for key in args.models:
        for seed in args.seeds:
            run_one(key, seed, args, classes, val_rows)
    summarise()


if __name__ == "__main__":
    main()
