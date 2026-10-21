"""
Step 17 - Extra checks that make the paper more complete (run AFTER rescore_plantdoc.py).
Running time: about 25-40 minutes (parts A-B on the CPU; parts C-D load every model once on the GPU).

Run from the project folder (plant-disease-fm):
    python src/extras.py              # all parts
    python src/extras.py --parts A B  # only some parts

A. Full-data reference on PlantDoc: a linear classifier trained on ALL 2,313 PlantDoc training
   photos (C by 3-fold CV on the training photos only), with and without the training photos
   whose labels the audit showed to be wrong. Puts the 16-photos-per-class result in context
   with OUR OWN full-data number instead of other papers' numbers.
B. Fair calibration: temperature scaling fitted on LAB images only, then ECE in the field. Removes
   the label-smoothing caveat (every model gets the same treatment, no field data used).
C. Prompt-template sensitivity: zero-shot accuracy for each of the 4 prompt templates separately
   (pre-specified style), PlantDoc 28-way, Plant Pathology 28-way, wheat 5-way.
D. Practical cost: parameters of the image encoder and images per second on this laptop's GPU.

Output: results/extras/full_data_plantdoc.csv, calibration_temperature.csv,
        prompt_templates.csv, model_cost.csv
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import fewshot as fs
import fm

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The least populated class")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

OUT = fm.ROOT / "results" / "extras"
AUDIT = fm.ROOT / "results" / "analysis" / "plantdoc_label_audit.csv"


def models_with(split):
    return [m for m in fm.MODELS if (fm.FEATURES / m / f"{split}.npz").exists()]


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return 100 * e


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------ A
def part_a():
    from rescore_plantdoc import key, versions
    all28, short = fm.class_list()
    short2label = {v: k for k, v in short.items()}
    audit = pd.read_csv(AUDIT) if AUDIT.exists() else None
    if audit is not None:
        audit["k"] = audit.path.map(key)
        bad = set(audit[(audit.split == "train") & (audit.category == "confirmed error")].k)
    else:
        bad = set()
        print("  (no label audit file - the 'clean pool' variant is skipped)")
    k16 = pd.read_csv(fm.ROOT / "results" / "fewshot" / "fewshot_summary.csv")
    rows = []
    for m in models_with("plantdoc_train_clean"):
        Xt, yt, pt = fs.load_feats(m, "plantdoc_test")
        Xp, yp, pp = fs.load_feats(m, "plantdoc_train_clean")
        V = versions(pt, yt, audit, short2label) if audit is not None else {"original": (np.ones(len(yt), bool), yt)}
        pools = {"all training photos": np.ones(len(yp), bool)}
        if bad:
            pools["audited errors removed"] = np.array([key(p) not in bad for p in pp])
        for pname, keep in pools.items():
            c, _ = fs.choose_c_cv(Xp[keep], yp[keep])
            pred = fs.logreg(Xp[keep], yp[keep], c).predict(Xt)
            for vname, (kt, y) in V.items():
                rows.append({"model": fm.MODELS[m]["label"], "training_pool": pname, "n_train": int(keep.sum()),
                             "C": c, "test_version": vname, "accuracy": 100 * np.mean(pred[kt] == y[kt])})
        g = k16[(k16.model == m) & (k16.method == "linear probe") & (k16.K == 16)]
        if not g.empty:
            rows.append({"model": fm.MODELS[m]["label"], "training_pool": "K=16 per class (reference)",
                         "n_train": 16 * 28, "C": None, "test_version": "original",
                         "accuracy": float(g.accuracy_mean.iloc[0])})
        print(f"  A {m}: done")
    df = pd.DataFrame(rows)
    df.round(2).to_csv(OUT / "full_data_plantdoc.csv", index=False)
    piv = df[df.test_version == "original"].pivot_table(index="model", columns="training_pool",
                                                        values="accuracy", sort=False)
    print("\nA. PlantDoc test accuracy (%), linear classifier on frozen features, original test labels")
    print(piv.round(1).to_string())
    if "K=16 per class (reference)" in piv:
        r = 100 * piv["K=16 per class (reference)"] / piv["all training photos"]
        print("   K=16 reaches this % of the full-data accuracy: " +
              ", ".join(f"{i}: {v:.0f}%" for i, v in r.dropna().items()))


# ------------------------------------------------------------------ B
def fit_temperature(logits, y_idx):
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    nll = [-np.mean(np.log(softmax(logits / t)[np.arange(len(y_idx)), y_idx] + 1e-12)) for t in grid]
    return float(grid[int(np.argmin(nll))])


def part_b():
    all28, _ = fm.class_list()
    lf = pd.read_csv(fm.ROOT / "results" / "fewshot" / "lab_to_field.csv")
    rows = []

    def add(name, lab_logits, lab_y, field):
        t = fit_temperature(lab_logits, lab_y)
        for data, (lg, y) in field.items():
            for tag, tt in [("before", 1.0), ("after", t)]:
                p = softmax(lg / tt)
                correct = (p.argmax(1) == y).astype(float)
                rows.append({"model": name, "data": data, "temperature": tt if tag == "after" else 1.0,
                             "scaling": tag, "accuracy": 100 * correct.mean(),
                             "mean_confidence": 100 * p.max(1).mean(), "ECE": ece(p.max(1), correct)})

    for m in models_with("plantvillage_train_sub"):
        lab_tr, lab_te = fs.load_feats(m, "plantvillage_train_sub"), fs.load_feats(m, "plantvillage_test_sub")
        g = lf[(lf.model == m) & (lf.lab_data == "sub")]
        c = float(g.C.iloc[0]) if not g.empty else fs.choose_c_cv(lab_tr[0], lab_tr[1])[0]
        clf = fs.logreg(lab_tr[0], lab_tr[1], c)
        cls = list(clf.classes_)
        idx = lambda y: np.array([cls.index(v) for v in y])
        field = {}
        for data, split in [("Lab (PlantVillage)", "plantvillage_test_sub"), ("PlantDoc", "plantdoc_test"),
                            ("Plant Pathology", "pp_test")]:
            d = fs.load_feats(m, split)
            if d is not None:
                field[data] = (clf.decision_function(d[0]), idx(d[1]))
        add(fm.MODELS[m]["label"], field["Lab (PlantVillage)"][0], field["Lab (PlantVillage)"][1], field)
        print(f"  B {m}: done")
    for m in fm.CNNS:
        p = fm.ROOT / "results" / "stats" / f"finetuned_{m}_logits.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        field = {data: (z[f"logits_{s}"], z[f"y_{s}"]) for data, s in
                 [("Lab (PlantVillage)", "plantvillage_test_sub"), ("PlantDoc", "plantdoc_test"),
                  ("Plant Pathology", "pp_test")] if f"logits_{s}" in z.files}
        add(fm.MODELS[m]["label"] + " (fine-tuned)", *field["Lab (PlantVillage)"], field)
    df = pd.DataFrame(rows)
    df.round(3).to_csv(OUT / "calibration_temperature.csv", index=False)
    piv = df[df.data != "Lab (PlantVillage)"].pivot_table(index="model", columns=["data", "scaling"],
                                                          values="ECE", sort=False)
    print("\nB. ECE (%) in the field before / after temperature scaling fitted on LAB images only")
    print(piv.round(1).to_string())


# ------------------------------------------------------------------ C + D
def part_cd(do_c, do_d):
    import torch
    all28, _ = fm.class_list()
    crow, drow = [], []
    rows_pd = fm.read_split("plantdoc_test")[:64]
    for m in fm.MODELS:
        if not (fm.FEATURES / m / "plantdoc_test.npz").exists():
            continue
        model, preprocess = fm.load_model(m)
        if do_c and fm.has_text_encoder(m):
            style = fm.MODELS[m]["prompt"]
            for data, split in [("PlantDoc 28-way", "plantdoc_test"), ("Plant Pathology 28-way", "pp_test"),
                                ("Wheat 5-way", "wheat_test")]:
                d = fs.load_feats(m, split)
                if d is None:
                    continue
                X, y, _ = d
                classes = all28 if "28" in data else sorted(set(y))
                with torch.inference_mode():
                    per_tpl = np.stack([fm.encode_texts(m, [tpl.format(fm.label_to_text(c, style))
                                                            for c in classes]) for tpl in fm.PROMPT_TEMPLATES])
                cls = np.array(classes)
                for i, tpl in enumerate(fm.PROMPT_TEMPLATES):
                    crow.append({"model": fm.MODELS[m]["label"], "data": data, "template": tpl,
                                 "accuracy": 100 * np.mean(cls[(X @ per_tpl[i].T).argmax(1)] == y)})
                w = per_tpl.mean(0)
                w /= np.linalg.norm(w, axis=1, keepdims=True)
                crow.append({"model": fm.MODELS[m]["label"], "data": data, "template": "average of 4 (used)",
                             "accuracy": 100 * np.mean(cls[(X @ w.T).argmax(1)] == y)})
        if do_d:
            enc = model.visual if fm.has_text_encoder(m) else model
            params = sum(p.numel() for p in enc.parameters()) / 1e6
            from torch.utils.data import DataLoader
            loader = DataLoader(fm.ImageCSV(rows_pd, preprocess), batch_size=fm.MODELS[m]["batch"])
            batches = [x for x in loader]
            with torch.inference_mode():
                f = model.encode_image if fm.has_text_encoder(m) else model
                f(batches[0].to(fm.DEVICE))                       # warm-up
                if fm.DEVICE == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for x in batches:
                    f(x.to(fm.DEVICE))
                if fm.DEVICE == "cuda":
                    torch.cuda.synchronize()
                ips = sum(len(x) for x in batches) / (time.time() - t0)
            drow.append({"model": fm.MODELS[m]["label"], "image_encoder_params_M": round(params, 1),
                         "images_per_second_gpu": round(ips, 1), "device": str(fm.DEVICE)})
        fm.free_model(m)
        print(f"  C/D {m}: done")
    if crow:
        df = pd.DataFrame(crow)
        df.round(2).to_csv(OUT / "prompt_templates.csv", index=False)
        s = df[df.template != "average of 4 (used)"].groupby(["model", "data"]).accuracy.agg(["min", "max"])
        s["used (average)"] = df[df.template == "average of 4 (used)"].set_index(["model", "data"]).accuracy
        print("\nC. Zero-shot accuracy (%): worst and best single template vs the 4-template average used")
        print(s.round(1).to_string())
    if drow:
        df = pd.DataFrame(drow)
        df.to_csv(OUT / "model_cost.csv", index=False)
        print("\nD. Image encoder size and speed on this laptop (feature extraction, 224 x 224)")
        print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if "A" in args.parts:
        part_a()
    if "B" in args.parts:
        part_b()
    if "C" in args.parts or "D" in args.parts:
        part_cd("C" in args.parts, "D" in args.parts)
    print("\nSaved: results/extras/")


if __name__ == "__main__":
    main()
