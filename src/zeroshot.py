"""
Step 5 - Zero-shot test: can a foundation model recognise field diseases from
text descriptions alone, without seeing a single training image?

Each model is tested with two prompt styles:
  common     : "a photo of a tomato leaf with late blight."
  scientific : "a photo of a Solanum lycopersicum (tomato) leaf infected by Phytophthora infestans (late blight)."
so that models trained on taxonomic text (BioCLIP) are compared fairly.

Run from the project folder (plant-disease-fm):
    python src/zeroshot.py                                # CLIP, BioCLIP, BioCLIP 2 on PlantDoc test
    python src/zeroshot.py --models bioclip2              # one model
    python src/zeroshot.py --data plantvillage_test       # lab images (needs Step 6 features, or is slow)

Output:
    results/zeroshot_summary.csv                                   one row per model x data x prompt style
    results/zeroshot/<model>_<data>_<style>_predictions.csv        per-image predictions
    results/zeroshot/<model>_<data>_<style>_per_class.csv          per-class accuracy
"""
import argparse
import csv
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import fm

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
RESULTS = fm.ROOT / "results"
STYLES = ["common", "scientific"]


def text_classifier(model_key, classes, style):
    weights = []
    for c in classes:
        t = fm.encode_texts(model_key, [tpl.format(fm.label_to_text(c, style))
                                        for tpl in fm.PROMPT_TEMPLATES])
        w = t.mean(axis=0)
        weights.append(w / np.linalg.norm(w))
    return np.stack(weights)


def evaluate(model_key, data, style, feats, labels, paths, classes, short):
    weights = text_classifier(model_key, classes, style)
    pred = [classes[i] for i in (feats @ weights.T).argmax(axis=1)]
    present = sorted(set(labels))  # metrics only over classes present in this test set

    out = RESULTS / "zeroshot"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{model_key}_{data}_{style}"
    with open(out / f"{tag}_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "correct"])
        for p, t, q in zip(paths, labels, pred):
            w.writerow([p, t, q, int(t == q)])
    with open(out / f"{tag}_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "short_name", "n", "accuracy"])
        for c in present:
            idx = [i for i, t in enumerate(labels) if t == c]
            w.writerow([c, short[c], len(idx), f"{100 * np.mean([pred[i] == c for i in idx]):.2f}"])

    return {
        "model": model_key, "model_name": fm.MODELS[model_key]["label"], "data": data,
        "prompt": style, "n_images": len(labels), "n_classes_eval": len(present),
        "accuracy": 100 * accuracy_score(labels, pred),
        "balanced_accuracy": 100 * balanced_accuracy_score(labels, pred),
        "macro_f1": 100 * f1_score(labels, pred, labels=present, average="macro", zero_division=0),
    }


def save_summary(new_rows):
    path = RESULTS / "zeroshot_summary.csv"
    fields = list(new_rows[0].keys())
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if "prompt" in r and r["prompt"]]
    keys = {(r["model"], r["data"], r["prompt"]) for r in new_rows}
    rows = [r for r in rows if (r["model"], r["data"], r["prompt"]) not in keys] + new_rows
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["clip_b16", "bioclip", "bioclip2"],
                    choices=[k for k in fm.MODELS if fm.has_text_encoder(k)])
    ap.add_argument("--data", default="plantdoc_test")
    args = ap.parse_args()

    print(f"Device: {fm.DEVICE}")
    classes, short = fm.class_list()
    results = []
    for m in args.models:
        feats, labels, paths = fm.encode_images(m, args.data)
        for style in STYLES:
            results.append(evaluate(m, args.data, style, feats, labels, paths, classes, short))
        fm.free_model(m)
    path = save_summary(results)

    print(f"\nZero-shot results on {args.data}")
    print(f"{'Model':<18}{'Prompt':<12}{'Acc %':>8}{'Bal.Acc %':>11}{'Macro-F1 %':>12}")
    for r in results:
        print(f"{r['model_name']:<18}{r['prompt']:<12}{r['accuracy']:>8.2f}"
              f"{r['balanced_accuracy']:>11.2f}{r['macro_f1']:>12.2f}")
    print(f"\nSaved: {path.relative_to(fm.ROOT)} and results/zeroshot/")


if __name__ == "__main__":
    main()
