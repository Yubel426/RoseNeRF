export CUDA_VISIBLE_DEVICES=5

SCENE=Pen
EXPERIMENT=logs_snerf_wo_warpnetloss
DATA_DIR=/home/nxt/nxtdg/datasets
CHECKPOINT_DIR=/home/nxt/nxtdg/logs/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/snerf_360.gin \
  --gin_bindings="Config.dataset_loader = 'llff'" \
  --gin_bindings="Config.factor = 0" \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr
