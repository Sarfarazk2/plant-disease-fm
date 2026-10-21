@echo off
REM Runs the whole study in order. Download PlantDoc, Plant Pathology 2020 and the JIC wheat images first
REM (see data\README.md). Every step can be re-run; feature extraction and fine-tuning resume where they stopped.
setlocal
set ALL_SPLITS=plantdoc_test plantdoc_train_clean plantvillage_test_sub plantvillage_train_sub pp_test pp_pool wheat_test wheat_pool

python src\check_setup.py || goto :error
python src\download_plantvillage.py || goto :error
python src\inspect_datasets.py || goto :error
python src\prepare_splits.py || goto :error
python src\make_lab_subsets.py || goto :error
python src\prepare_plantpathology.py || goto :error
python src\prepare_wheat.py || goto :error
python src\extract_features.py --models clip_b16 bioclip bioclip2 clip_l14 siglip_b16 dinov2_b14 dinov2_l14 resnet50 effnet_b0 mobilenet_v3 --splits %ALL_SPLITS% --workers 2 || goto :error
python src\zeroshot.py || goto :error
python src\fewshot.py || goto :error
REM Optional robustness check (all 43,596 lab images, CLIP only; extraction about 1.5 h):
REM   python src\extract_features.py --models clip_b16 --splits plantvillage_train plantvillage_test --workers 2
REM   python src\fewshot.py --models clip_b16 --lab full
python src\eval_plantpathology.py || goto :error
python src\fusion.py || goto :error
python src\eval_wheat.py || goto :error
python src\openset.py || goto :error
python src\finetune_cnn.py --model resnet50 || goto :error
python src\finetune_cnn.py --model effnet_b0 || goto :error
python src\finetune_cnn.py --model mobilenet_v3 || goto :error
python src\stats.py || goto :error
python src\analysis.py || goto :error
python src\rescore_plantdoc.py || goto :error
python src\extras.py || goto :error
python src\make_figures.py || goto :error
echo.
echo All steps finished.
goto :eof
:error
echo.
echo A step failed - see the message above. Fix it and run this file again (finished steps are fast).
exit /b 1
