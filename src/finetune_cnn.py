"""
Step 13 - Classic baseline: FULLY fine-tuned ResNet-50 (ImageNet) on lab images, tested in the field.
This is the standard approach of most plant-disease papers (e.g. the original PlantDoc paper).

LONG RUN: about 45-75 minutes on a 4 GB laptop GPU. Safe to stop: a checkpoint is saved after
every epoch and the same command continues from the last finished epoch.

Run from the project folder (plant-disease-fm):
    python src/finetune_cnn.py                     # ResNet-50, 8 epochs
    python src/finetune_cnn.py --epochs 5          # shorter
    python src/finetune_cnn.py --model effnet_b0   # EfficientNet-B0 instead

Training data : PlantVillage balanced lab subset (200 images x 28 classes), augmented.
Model selection: best epoch on the PlantVillage lab test subset (no field data used).
Evaluation    : PlantDoc test (28-way), Plant Pathology test (28-way and 3-way apple).

Output: models/<model>_lab_last.pt (checkpoint), results/finetune/finetune_<model>.csv
"""
import argparse
import csv
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

import fm

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
OUT = fm.ROOT / "results" / "finetune"
CKPT = fm.ROOT / "models"
APPLE = ["Apple___Apple_scab", "Apple___Cedar_apple_rust", "Apple___healthy"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class LabeledCSV(Dataset):
    def __init__(self, rows, classes, tf):
        self.rows, self.idx, self.tf = rows, {c: i for i, c in enumerate(classes)}, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        r = self.rows[i]
        with Image.open(fm.ROOT / r["path"]) as im:
            return self.tf(im.convert("RGB")), self.idx[r["label"]]


def metrics(y, p, present):
    return {"accuracy": 100 * accuracy_score(y, p),
            "balanced_accuracy": 100 * balanced_accuracy_score(y, p),
            "macro_f1": 100 * f1_score(y, p, labels=present, average="macro", zero_division=0)}


@torch.inference_mode()
def predict(model, rows, classes, tf, bs, workers):
    loader = DataLoader(LabeledCSV(rows, classes, tf), batch_size=bs, num_workers=workers)
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(fm.DEVICE)).float().cpu())
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=fm.CNNS)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)        # fits a 4 GB GPU
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import timm
    torch.manual_seed(0)
    np.random.seed(0)
    classes, _ = fm.class_list()
    cfg = fm.MODELS[args.model]
    model = timm.create_model(cfg["name"], pretrained=True, num_classes=len(classes)).to(fm.DEVICE)

    train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                          T.RandomVerticalFlip(), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(),
                          T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])

    lab_tr = fm.read_split("plantvillage_train_sub")
    lab_te = fm.read_split("plantvillage_test_sub")
    loader = DataLoader(LabeledCSV(lab_tr, classes, train_tf), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    CKPT.mkdir(exist_ok=True)
    last, best_path = CKPT / f"{args.model}_lab_last.pt", CKPT / f"{args.model}_lab_best.pt"
    start, best_acc = 0, -1.0
    if last.exists():
        ck = torch.load(last, map_location=fm.DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best_acc = ck["epoch"] + 1, ck["best_acc"]
        print(f"Resuming after epoch {start} (best lab accuracy so far {best_acc:.1f}%)")

    for ep in range(start, args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tqdm(loader, desc=f"epoch {ep + 1}/{args.epochs}"):
            x, y = x.to(fm.DEVICE), y.to(fm.DEVICE)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        model.eval()
        lg, yy = predict(model, lab_te, classes, test_tf, 32, args.workers)
        acc = 100 * np.mean(lg.argmax(1) == yy)
        print(f"  epoch {ep + 1}: loss {tot / n:.3f} | lab accuracy {acc:.1f}% | {(time.time() - t0) / 60:.1f} min")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best_acc": best_acc}, last)

    # ---- evaluate the best (lab-selected) epoch
    model.load_state_dict(torch.load(best_path, map_location=fm.DEVICE, weights_only=True))
    model.eval()
    results = []
    for name, split in [("PlantVillage lab (sub)", "plantvillage_test_sub"),
                        ("PlantDoc field", "plantdoc_test"), ("Plant Pathology field", "pp_test")]:
        rows = fm.read_split(split)
        if not rows:
            continue
        lg, yy = predict(model, rows, classes, test_tf, 32, args.workers)
        present = sorted(set(yy.tolist()))
        results.append({"model": args.model, "data": name, "setting": "28-way",
                        **metrics(yy, lg.argmax(1), present)})
        if split == "pp_test":
            ai = [classes.index(c) for c in APPLE]
            p3 = np.array(ai)[lg[:, ai].argmax(1)]
            results.append({"model": args.model, "data": name, "setting": "3-way",
                            **metrics(yy, p3, present)})

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"finetune_{args.model}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nFully fine-tuned {cfg['label']} (trained on lab images only, best epoch by lab accuracy)")
    for r in results:
        print(f"  {r['data']:<24} {r['setting']:<7} acc {r['accuracy']:5.1f} | bal.acc {r['balanced_accuracy']:5.1f}"
              f" | macro-F1 {r['macro_f1']:5.1f}")
    print(f"Saved: results/finetune/finetune_{args.model}.csv")


if __name__ == "__main__":
    main()
