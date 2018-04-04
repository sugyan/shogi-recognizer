#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
python scripts/retrain.py \
    --image_dir ${BASEDIR}/dataset \
    --architecture mobilenet_1.0_128 \
    --random_crop 5 \
    --random_scale 5 \
    --random_brightness 5 \
    --how_many_training_steps 1000
