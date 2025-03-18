#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")

TRAIN_SCRIPT_PATH="$SCRIPT_DIR/train_pretrain_deepspeed.py"
DS_CONFIG_PATH="$SCRIPT_DIR/llm_learn/ds_config.json"

deepspeed --num_gpus 2 "$TRAIN_SCRIPT_PATH" \
--deepspeed_config "$DS_CONFIG_PATH"