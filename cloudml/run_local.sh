#!/bin/bash

root_dir=$(cd $(dirname $0)/.. && pwd)
export PYTHONPATH=${PYTHONPATH}:${root_dir}/models/research/slim

TRAINER_PACKAGE_PATH=${root_dir}/trainer
MAIN_TRAINER_MODULE="trainer.task"
JOB_DIR=${root_dir}/logdir

cd ${root_dir}/trainer
gcloud ml-engine local train \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --job-dir $JOB_DIR \
    -- \
    --number_of_steps 3
