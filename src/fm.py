"""
fm.py - shared code for all foundation-model experiments.

  * MODELS            : the models compared in the paper
  * load_model()      : loads a model (downloads it once, then uses the local copy)
  * encode_images()   : computes image features for a split CSV and CACHES them in
                        features/<model>/<split>.npz. Work is saved every 2,048 images,
                        so an interrupted run continues where it stopped.
  * encode_texts()    : text features (CLIP-style models only, used for zero-shot)
  * label_to_text()   : turns a PlantVillage label into English or scientific text
"""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
FEATURES = ROOT / "features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 2048  # images per saved chunk (resume point)

# "openclip" models have image AND text encoders (zero-shot possible);
# "timm" models are image-only (used for few-shot / linear-probe comparisons).
# "prompt" = the text style used in the main experiments, FIXED IN ADVANCE from each model's
# training text (not chosen from test results): web-trained models -> common names,
# models trained on taxonomic text (BioCLIP family) -> scientific names.
MODELS = {
    # OpenAI CLIP weights were trained with QuickGELU, so the "-quickgelu" config must be used
    "clip_b16":   {"kind": "openclip", "name": "ViT-B-16-quickgelu", "pretrained": "openai",
                   "label": "CLIP ViT-B/16", "batch": 32, "prompt": "common"},
    "bioclip":    {"kind": "openclip", "name": "hf-hub:imageomics/bioclip", "pretrained": None,
                   "label": "BioCLIP", "batch": 32, "prompt": "scientific"},
    "bioclip2":   {"kind": "openclip", "name": "hf-hub:imageomics/bioclip-2", "pretrained": None,
                   "label": "BioCLIP 2", "batch": 16, "prompt": "scientific"},
    # CONTROL for BioCLIP 2: the exact general-purpose checkpoint BioCLIP 2 was initialised from
    # (same ViT-L/14 architecture, same LAION-2B pre-training) -> the only difference is the
    # biology training. Same logic as the CLIP ViT-B/16 (OpenAI) -> BioCLIP pair.
    "clip_l14":   {"kind": "openclip", "name": "ViT-L-14", "pretrained": "laion2b_s32b_b82k",
                   "label": "CLIP ViT-L/14 (LAION-2B)", "batch": 16, "prompt": "common"},
    "siglip_b16": {"kind": "openclip", "name": "ViT-B-16-SigLIP", "pretrained": "webli",
                   "label": "SigLIP ViT-B/16", "batch": 32, "prompt": "common"},
    # DINOv2 is run at 224x224 (instead of its default 518) so all models see the same resolution
    "dinov2_b14": {"kind": "timm", "name": "vit_base_patch14_dinov2.lvd142m", "pretrained": None,
                   "label": "DINOv2 ViT-B/14", "batch": 32, "img_size": 224},
    # SIZE-MATCHED CONTROLS (ViT-L, like BioCLIP 2):
    # BioCLIP 2 was initialised from exactly this LAION-2B CLIP ViT-L/14 checkpoint, so the
    # pair (clip_l14 -> bioclip2) isolates the effect of the biology training data.
    # (Same idea at ViT-B size: clip_b16 -> bioclip.)  Web-trained -> common-name prompts.
    "clip_l14":   {"kind": "openclip", "name": "ViT-L-14", "pretrained": "laion2b_s32b_b82k",
                   "label": "CLIP ViT-L/14 (LAION-2B)", "batch": 16, "prompt": "common"},
    # optional: large self-supervised image model (is DINOv2 only behind because it is smaller?)
    "dinov2_l14": {"kind": "timm", "name": "vit_large_patch14_dinov2.lvd142m", "pretrained": None,
                   "label": "DINOv2 ViT-L/14", "batch": 16, "img_size": 224},
    # Conventional CNN baselines (ImageNet-1k supervised), the standard in plant-disease papers
    "resnet50":   {"kind": "timm", "name": "resnet50.a1_in1k", "pretrained": None,
                   "label": "ResNet-50 (ImageNet)", "batch": 32, "cnn": True},
    "effnet_b0":  {"kind": "timm", "name": "efficientnet_b0.ra_in1k", "pretrained": None,
                   "label": "EfficientNet-B0 (ImageNet)", "batch": 32, "cnn": True},
    "mobilenet_v3": {"kind": "timm", "name": "mobilenetv3_large_100.ra_in1k", "pretrained": None,
                     "label": "MobileNetV3-L (ImageNet)", "batch": 32, "cnn": True},
}
FOUNDATION = [k for k, v in MODELS.items() if not v.get("cnn")]
CNNS = [k for k, v in MODELS.items() if v.get("cnn")]


# ---------------------------------------------------------------- data
def read_split(name, shared_only=True):
    """Read data/splits/<name>.csv -> list of dict rows (path, label, short_name ...)."""
    with open(SPLITS / f"{name}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if shared_only and rows and "shared" in rows[0]:
        rows = [r for r in rows if r["shared"] == "1"]
    return rows


def class_list():
    """The 28 shared classes, in a fixed order (PlantVillage labels) + short names."""
    with open(SPLITS / "class_mapping.csv", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["label"])
    return [r["label"] for r in rows], {r["label"]: r["short_name"] for r in rows}


class ImageCSV(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(ROOT / self.rows[i]["path"]) as im:
            return self.transform(im.convert("RGB"))


# ---------------------------------------------------------------- models
_cache = {}


def load_model(key):
    """Returns (model, image_transform). Cached in memory."""
    if key in _cache:
        return _cache[key]
    cfg = MODELS[key]
    print(f"Loading {cfg['label']} (first time: downloads the weights) ...")
    if cfg["kind"] == "openclip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["name"], pretrained=cfg["pretrained"])
    else:
        import timm
        extra = {"img_size": cfg["img_size"]} if "img_size" in cfg else {}
        model = timm.create_model(cfg["name"], pretrained=True, num_classes=0, **extra)
        data_cfg = timm.data.resolve_model_data_config(model)
        data_cfg["input_size"] = (3, 224, 224)   # same input resolution for every model
        preprocess = timm.data.create_transform(**data_cfg, is_training=False)
    model = model.eval().to(DEVICE)
    _cache[key] = (model, preprocess)
    return _cache[key]


def free_model(key):
    """Release GPU memory before loading the next model."""
    _cache.pop(key, None)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def has_text_encoder(key):
    return MODELS[key]["kind"] == "openclip"


def _model_id(key):
    return f"{MODELS[key]['name']}|{MODELS[key]['pretrained']}|224"


@torch.inference_mode()
def _compute(key, rows, pbar, num_workers=0):
    model, preprocess = load_model(key)
    loader = DataLoader(ImageCSV(rows, preprocess), batch_size=MODELS[key]["batch"],
                        shuffle=False, num_workers=num_workers)
    feats = []
    for x in loader:
        x = x.to(DEVICE)
        f = model.encode_image(x) if has_text_encoder(key) else model(x)
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
        pbar.update(len(x))
    return torch.cat(feats).numpy().astype(np.float32)


def _load_npz(path, model_id, paths):
    if not path.exists():
        return None
    try:
        d = np.load(path, allow_pickle=True)
        if "model_id" in d.files and str(d["model_id"]) == model_id and list(d["paths"]) == paths:
            return d["feats"]
    except Exception:
        pass
    return None


def encode_images(key, split, shared_only=True, num_workers=0):
    """Image features for every image in data/splits/<split>.csv (L2-normalised).
    Returns (features [N, D] float32, labels list, paths list). Resumable."""
    out = FEATURES / key / f"{split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = read_split(split, shared_only)
    paths = [r["path"] for r in rows]
    labels = [r["label"] for r in rows]
    mid = _model_id(key)

    feats = _load_npz(out, mid, paths)
    if feats is not None:
        return feats, labels, paths

    # A "_sub" split is a subset of the full split: reuse full features if they already exist
    if split.endswith("_sub"):
        full = FEATURES / key / f"{split[:-4]}.npz"
        if full.exists():
            d = np.load(full, allow_pickle=True)
            if "model_id" in d.files and str(d["model_id"]) == mid:
                index = {p: i for i, p in enumerate(d["paths"])}
                if all(p in index for p in paths):
                    feats = d["feats"][[index[p] for p in paths]]
                    np.savez_compressed(out, feats=feats, labels=np.array(labels, dtype=object),
                                        paths=np.array(paths, dtype=object), model_id=np.array(mid))
                    print(f"  {key} | {split}: taken from the full {split[:-4]} features")
                    return feats, labels, paths

    chunks =[rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    done = [None] * len(chunks)
    for ci, ch in enumerate(chunks):
        done[ci] = _load_npz(out.with_name(f"{split}.part{ci:03d}.npz"), mid,
                             [r["path"] for r in ch])
    n_done = sum(len(ch) for ch, d in zip(chunks, done) if d is not None)
    if n_done:
        print(f"  resuming {key} | {split}: {n_done}/{len(rows)} images already done")

    with tqdm(total=len(rows), initial=n_done, desc=f"{key} | {split}", unit="img") as pbar:
        for ci, ch in enumerate(chunks):
            if done[ci] is not None:
                continue
            f = _compute(key, ch, pbar, num_workers)
            np.savez_compressed(out.with_name(f"{split}.part{ci:03d}.npz"), feats=f,
                                paths=np.array([r["path"] for r in ch], dtype=object),
                                model_id=np.array(mid))
            done[ci] = f

    feats = np.concatenate(done) if done else np.zeros((0, 1), np.float32)
    np.savez_compressed(out, feats=feats, labels=np.array(labels, dtype=object),
                        paths=np.array(paths, dtype=object), model_id=np.array(mid))
    for ci in range(len(chunks)):
        out.with_name(f"{split}.part{ci:03d}.npz").unlink(missing_ok=True)
    return feats, labels, paths


@torch.inference_mode()
def encode_texts(key, texts):
    import open_clip
    model, _ = load_model(key)
    tokenizer = open_clip.get_tokenizer(MODELS[key]["name"])
    f = model.encode_text(tokenizer(texts).to(DEVICE))
    return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()


# ---------------------------------------------------------------- text prompts
_CROP = {"Corn_(maize)": "corn", "Pepper,_bell": "bell pepper",
         "Cherry_(including_sour)": "cherry"}
_CROP_SCI = {
    "apple": "Malus domestica", "blueberry": "Vaccinium corymbosum", "cherry": "Prunus avium",
    "corn": "Zea mays", "grape": "Vitis vinifera", "peach": "Prunus persica",
    "bell pepper": "Capsicum annuum", "potato": "Solanum tuberosum", "raspberry": "Rubus idaeus",
    "soybean": "Glycine max", "squash": "Cucurbita pepo", "strawberry": "Fragaria x ananassa",
    "tomato": "Solanum lycopersicum", "wheat": "Triticum aestivum",
}
_DISEASE = {
    "Cercospora_leaf_spot Gray_leaf_spot": "gray leaf spot",
    "Common_rust_": "common rust",
    "Northern_Leaf_Blight": "northern leaf blight",
    "Spider_mites Two-spotted_spider_mite": "two-spotted spider mite damage",
    "Tomato_Yellow_Leaf_Curl_Virus": "yellow leaf curl virus",
    "Tomato_mosaic_virus": "mosaic virus",
    "Apple_scab": "apple scab",
    "Cedar_apple_rust": "cedar apple rust",
}
# causal organism of each disease (standard plant-pathology names)
_PATHOGEN = {
    "apple scab": "Venturia inaequalis",
    "cedar apple rust": "Gymnosporangium juniperi-virginianae",
    "gray leaf spot": "Cercospora zeae-maydis",
    "common rust": "Puccinia sorghi",
    "northern leaf blight": "Exserohilum turcicum",
    "black rot": "Guignardia bidwellii",
    "bacterial spot": "Xanthomonas",
    "early blight": "Alternaria solani",
    "late blight": "Phytophthora infestans",
    "powdery mildew": "Podosphaera xanthii",
    "leaf mold": "Passalora fulva",
    "septoria leaf spot": "Septoria lycopersici",
    "two-spotted spider mite damage": "Tetranychus urticae",
    "yellow leaf curl virus": "tomato yellow leaf curl virus",
    "mosaic virus": "tomato mosaic virus",
}
# crop-specific pathogens (checked first; the same disease name can have a different cause per crop)
_PATHOGEN_CROP = {
    ("wheat", "powdery mildew"): "Blumeria graminis f. sp. tritici",
    ("wheat", "yellow rust"): "Puccinia striiformis f. sp. tritici",
    ("wheat", "brown rust"): "Puccinia triticina",
    ("wheat", "septoria leaf blotch"): "Zymoseptoria tritici",
}


def _pathogen(crop, disease):
    return _PATHOGEN_CROP.get((crop, disease), _PATHOGEN.get(disease, disease))


def _parse(label):
    crop, _, disease = label.partition("___")
    crop = _CROP.get(crop, crop.replace("_", " ")).lower()
    disease = disease if disease == "healthy" else _DISEASE.get(disease, disease.replace("_", " ")).lower()
    return crop, disease


def label_to_text(label, style="common"):
    """style='common'     -> 'tomato leaf with late blight'
       style='scientific' -> 'Solanum lycopersicum (tomato) leaf infected by Phytophthora infestans (late blight)'"""
    crop, disease = _parse(label)
    if style == "scientific":
        sci = _CROP_SCI.get(crop, crop)
        if disease == "healthy":
            return f"healthy {sci} ({crop}) leaf"
        return f"{sci} ({crop}) leaf infected by {_pathogen(crop, disease)} ({disease})"
    if disease == "healthy":
        return f"healthy {crop} leaf"
    return f"{crop} leaf with {disease}"


PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a field photo of a {}.",
    "a plant disease photo: {}.",
]
