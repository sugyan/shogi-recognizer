#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
python scripts/retrain.py \
    --image_dir ${BASEDIR}/dataset \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/classification/1 \
    --how_many_training_steps 30000
