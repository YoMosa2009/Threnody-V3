#!/usr/bin/env bash
# Copies each SFT checkpoint's adapter to sft_v2_keep/ before the trainer rotates it out (save_total_limit),
# so checkpoint averaging can use them. Runs detached while train_sft.py is alive.
D=${DRIVE:-/content/drive/MyDrive/Threnody}; K=$D/sft_v2_keep; mkdir -p $K
while pgrep -f train_sft >/dev/null; do
  for c in $D/sft_v2/checkpoint-*; do n=$(basename $c); [ -f $K/$n/adapter_model.safetensors ] && continue
    sleep 60; mkdir -p $K/$n && cp $c/adapter_model.safetensors $c/adapter_config.json $c/trainer_state.json $K/$n/ 2>/dev/null; done
  sleep 120; done
