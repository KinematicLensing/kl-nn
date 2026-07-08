#!/bin/bash

MODEL_ROOT="/ocean/projects/phy250048p/shared/models"
MODEL_NAME=${MODEL_NAME:-"ViT-CNN-flow_tf_train"}
KEEP_EPOCHS=${KEEP_EPOCHS:-"199"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-"200"}

# Delete all epochs except the ones specified in KEEP_EPOCHS
for epoch in $(seq 0 $((TOTAL_EPOCHS - 1))); do
    if [[ ! " ${KEEP_EPOCHS} " =~ " ${epoch} " ]]; then
        echo "Deleting epoch $epoch for model $MODEL_NAME"
        rm -rf "$MODEL_ROOT/$MODEL_NAME/$MODEL_NAME$epoch"
    else
        echo "Keeping epoch $epoch for model $MODEL_NAME"
    fi
done