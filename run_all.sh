#!/usr/bin/env bash
set -e

# Runs the whole study in order. Download PlantDoc, Plant Pathology 2020 and the JIC wheat images first
# (see data\README.md). Every step can be re-run; feature extraction and fine-tuning resume where they stopped.

ALL_SPLITS="plantdoc_test plantdoc_train_clean plantvillage_test_sub plantvillage_train_sub pp_test pp_pool wheat_test wheat_pool"

python src/check_setup.py || exit 1
python src/download_plantvillage.py || exit 1
python src/inspect_datasets.py || exit 1
python src/prepare_splits.py || exit 1
python src/make_lab_subsets.py || exit 1
python src/prepare_plantpathology.py || exit 1
python src/prepare_wheat.py || exit 1
python src/extract_features.py --models clip_b16 bioclip bioclip2 clip_l14 siglip_b16 dinov2_b14 dinov2_l14 resnet50 effnet_b0 mobilenet_v3 --splits $ALL_SPLITS --workers 2 || exit 1
python src/zeroshot.py || exit 1
python src/fewshot.py || exit 1
# Optional robustness check (all 43,596 lab images, CLIP only; extraction about 1.5 h):
#   python src/extract_features.py --models clip_b16 --splits plantvillage_train plantvillage_test --workers 2
#   python src/fewshot.py --models clip_b16 --lab full
python src/eval_plantpathology.py || exit 1
python src/fusion.py || exit 1
python src/eval_wheat.py || exit 1
python src/openset.py || exit 1
python src/finetune_cnn.py --model resnet50 || exit 1
python src/finetune_cnn.py --model effnet_b0 || exit 1
python src/finetune_cnn.py --model mobilenet_v3 || exit 1
python src/stats.py || exit 1
python src/analysis.py || exit 1
python src/rescore_plantdoc.py || exit 1
python src/extras.py || exit 1
python src/make_figures.py || exit 1
echo
echo All steps finished.
