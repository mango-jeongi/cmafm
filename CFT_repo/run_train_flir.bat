@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
set WANDB_DISABLED=true
set WANDB_MODE=disabled

cd /d "d:\★RGB-LWIR(멘토ver-최종)\CFT_repo"

python train.py ^
  --cfg models/transformer/yolov5l_cmafm_M3FD.yaml ^
  --data data/multispectral/M3FD_FLIR.yaml ^
  --hyp data/hyp.scratch.yaml ^
  --epochs 30 ^
  --batch-size 4 ^
  --img-size 640 ^
  --project runs/train ^
  --name cmafm_m3fd_flir ^
  --exist-ok ^
  --workers 2 ^
  2>&1 | tee runs\train\cmafm_m3fd_flir_log.txt
