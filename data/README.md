# Datasets

The datasets are not included in this repository. Download each one into the folder shown, then run
the preparation scripts (`run_all.bat` / `run_all.sh` do this in order). Please check and respect each
dataset's own licence and terms of use on its source page.

| Dataset | Use in the paper | Source | Put it in |
|---|---|---|---|
| PlantVillage (colour images, official leaf-grouped split) | lab training source | Hugging Face dataset `mohanty/PlantVillage` — downloaded automatically by `src/download_plantvillage.py` | `data/plantvillage/` (created by the script) |
| PlantDoc | field test 1 + few-shot pool | https://github.com/pratikkayal/PlantDoc-Dataset (download or clone; keep the `train/` and `test/` folders) | `data/plantdoc/PlantDoc-Dataset-master/` |
| Plant Pathology 2020 (FGVC7) | field test 2 (apple orchard) | https://www.kaggle.com/c/plant-pathology-2020-fgvc7 (Kaggle account needed; extract the ZIP so that `train.csv` and `images/` are inside) | `data/plantpathology/` |
| JIC wheat disease images (999 photos, 5 classes) | new-crop test, unknown-crop set | https://zenodo.org/record/7573133 (released by Long *et al.*, *Plant Pathology* 72(3), 2023) | `data/wheat_jic/` (any sub-folder depth; the label is read from the file-name prefix, e.g. `BrownRust91.jpg`) |

After preparation, `data/splits/` holds every split and few-shot file used in the paper (fixed seeds).
