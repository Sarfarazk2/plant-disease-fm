# Lab-to-field plant disease recognition with biology foundation models

Code, splits and label audit for the paper
**"Laboratory Accuracy Is Not Field Accuracy: A Controlled Evaluation of Biology Foundation Models for Plant Disease Recognition Under Domain Shift"** (S. K. Khalil et al., under review, 2026).

Models trained on clean laboratory leaf photos (PlantVillage) fail on real field photos. This
repository compares 13 image encoders — biology vision-language models (BioCLIP, BioCLIP 2), web
vision-language models (CLIP, SigLIP), self-supervised models (DINOv2) and ImageNet CNNs (frozen and
fully fine-tuned) — on three field datasets, with zero-shot text prompts, lab-trained classifiers,
few-shot learning (1–16 field photos per class) and a text + image fusion rule (v2).

**Main findings**

| Finding | Evidence |
|---|---|
| BioCLIP 2 transfers from lab to field far better than all other models | 57.6% (PlantDoc) and 65.9% (Plant Pathology) when trained on lab photos only; ahead of all 12 other configurations, at least 22 points above every fine-tuned CNN run |
| The gain comes from biology training, not model size | +20.8 / +27.7 points over CLIP ViT-L/14 (LAION-2B), the exact checkpoint BioCLIP 2 was initialised from; DINOv2-L is no better than DINOv2-B |
| Lab accuracy does not predict field accuracy | 13 models within 3.5 points in the lab spread over 33–47 points in the field (rank correlation 0.36 and 0.26, n.s.); fine-tuned CNN checkpoints vary by up to 27 field points at near-constant lab accuracy |
| PlantDoc contains verifiable label errors | ≥ 10 of 236 test and ≥ 107 of 2,342 training images contradict their own original file names (see `label_audit/`) |
| Reliability-weighted text + image fusion (v2) is robust | best mean accuracy and smallest worst-case loss of 8 rules over 60 settings; 8.1 points above Tip-Adapter |
| Lab photos help only when field photos are scarce | +11.3 points at 1 field photo per class with balanced weighting, +0.6 at 16 |

All settings (classifier strength, fusion weights, prompt style) are chosen **without field data**.

---

## 1. Setup

Tested on Windows 10, Python 3.11, PyTorch 2.14.0 + CUDA 12.6, NVIDIA Quadro M1200 (4 GB).
Any CUDA GPU with ≥ 4 GB works; a CPU works too, but feature extraction is much slower.

```bash
conda create -n plantfm python=3.11 -y
conda activate plantfm
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python src/check_setup.py
```

`requirements-lock.txt` lists the exact versions of every package used for the paper.

## 2. Data

The datasets are **not** redistributed here. Download them into `data/` as described in
[`data/README.md`](data/README.md). The train / test / few-shot splits used in the paper are
regenerated deterministically by the preparation scripts (fixed seeds).

## 3. Run everything

`run_all.bat` (Windows) or `run_all.sh` (Linux/macOS) runs every step in order. Approximate times on
the 4 GB laptop GPU above:

| Step | Script | What it does | Time |
|---|---|---|---|
| 1 | `download_plantvillage.py` | PlantVillage with the official leaf-grouped split | download |
| 2 | `inspect_datasets.py` | counts, corrupt-image check | minutes |
| 3 | `prepare_splits.py` | PlantDoc ↔ PlantVillage class mapping, duplicate removal, few-shot sets | minutes |
| 4 | `make_lab_subsets.py` | balanced lab subsets (200 / 100 per class) | seconds |
| 5 | `prepare_plantpathology.py`, `prepare_wheat.py` | field datasets 2 and 3 | minutes |
| 6 | `extract_features.py` | frozen features for every model and split (resumable) | 20 min (CNNs) – 85 min (ViT-L) per model |
| 7 | `zeroshot.py`, `fewshot.py` | zero-shot, lab-to-field, few-shot, 5-way episodes | 15 min |
| 8 | `eval_plantpathology.py`, `fusion.py`, `eval_wheat.py`, `openset.py` | apple, fusion v2, wheat, unknown detection | 40 min |
| 9 | `finetune_cnn.py --model <cnn>` | full fine-tuning of each CNN on lab photos (resumable) | 21–64 min each |
| 10 | `stats.py` | bootstrap CIs, paired permutation tests, controlled pairs | 30 min |
| 11 | `analysis.py`, `rescore_plantdoc.py`, `extras.py` | errors, calibration, label audit, extra checks | 45 min |
| 12 | `make_figures.py` | final figures (SVG/PDF/EPS/PNG/TIFF + CSV) and tables (CSV/LaTeX/Excel) | 2 min |

Outputs: `results/` (all numbers as CSV), `figures/final/` (paper figures), `results/paper_tables/`.

### Revision experiments (reviewer requests)

`run_revision.bat` (or notebook `09_revision_experiments`) runs the additional experiments of the revised paper;
results go to `results/revision/` and never overwrite the original results:

| Script | What it does | Time |
|---|---|---|
| `revision_baselines.py` | Tip-Adapter (default and support-tuned) vs fusion v2; laboratory + field training (pooled and balanced) | ~1 h, CPU |
| `extract_features.py --models bioclip2 dinov2_b14 --splits plantvillage_train plantvillage_test` | features of all PlantVillage images | ~5-6 h, GPU |
| `revision_fulllab.py` | laboratory classifier on all 30,978 lab images vs the 200-per-class subset | ~10 min, CPU |
| `revision_finetune.py` | fine-tuned CNNs with a lab validation set, 3 seeds, field accuracy logged per epoch | ~8 h, GPU |
| `make_sample_figure.py` | example images from the four datasets (paper Fig. 2) | seconds |

### Running the study in Jupyter notebooks (VS Code)

`notebooks/` runs exactly the same steps one at a time, with explanations and the results shown after
each step: `00_START_HERE` → `08_compare_with_paper`. The notebooks call the scripts in `src/`, so the results
are the same as with `run_all`. `08_compare_with_paper` compares every number in your `results/` with
`reference_results/` (the result files used for the paper). One-time setup: install VS Code with the *Python* and
*Jupyter* extensions, `pip install ipykernel` inside the `plantfm` environment, open the project folder and pick
the `plantfm` kernel at the top right of a notebook.

## 4. Repository layout

```
notebooks/      step-by-step Jupyter version of run_all (00-08)
reference_results/  result files used in the paper (for comparison)
src/            all scripts (fm.py = model list, feature cache, prompts; figstyle.py = figure style)
data/README.md  where to download each dataset
label_audit/    PlantDoc label audit (see below)
results/        created by the scripts
figures/        created by the scripts
```

## 5. PlantDoc label audit

`label_audit/plantdoc_label_audit.csv` lists PlantDoc images whose label is contradicted by the
image's own original web file name (e.g. `corn-gray-leaf-spot-f4.jpg` filed under northern leaf
blight), plus test images flagged by model consensus and checked visually. Columns: split, path,
source of evidence, category (confirmed error / probable error / questionable), corrected label
(test images, when the true class is one of the 28 classes) and the evidence.
`plantdoc_filename_audit.csv` lists every file-name keyword hit with its judgement (including false
hits), and `label_audit_candidates_reviewed.csv` the 24 model-flagged test images.
The audit is a lower bound: most file names carry no disease information.

## 6. Licence and citation

Code: MIT licence (see `LICENSE`). Datasets keep their own licences (see `data/README.md`).
If you use this code or the label audit, please cite the paper (`CITATION.cff`).
