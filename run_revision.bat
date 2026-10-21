@echo off
REM Revision experiments (reviewer requests). Run from the project folder inside the plantfm environment.
REM Every step can be stopped and re-run: it continues where it stopped.
REM Order: R3 (CPU, ~1 h)  ->  feature extraction (GPU, ~5-6 h)  ->  R2 (CPU, ~1 h)  ->  R1 (GPU, ~6-7 h)
setlocal

echo === R3: Tip-Adapter baseline + lab+field training (CPU, about 1 hour) ===
python src\revision_baselines.py || goto :error

echo === Full PlantVillage features for BioCLIP 2 and DINOv2-B (GPU, about 5-6 hours) ===
python src\extract_features.py --models bioclip2 dinov2_b14 --splits plantvillage_train plantvillage_test --workers 2 || goto :error

echo === R2: lab classifier on all PlantVillage images (CPU, about 1 hour) ===
python src\revision_fulllab.py || goto :error

echo === R1: fine-tuned CNNs with lab validation set, 3 seeds (GPU, about 6-7 hours) ===
python src\revision_finetune.py || goto :error

echo.
echo All revision experiments finished. Zip the folder results\revision and send it.
goto :eof
:error
echo.
echo A step failed - see the message above. Send the last 20 lines. Re-running this file skips finished work.
exit /b 1
