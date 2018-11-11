#!/bin/bash

if [ -z "${BUCKET_NAME}" ]; then
    echo "BUCKET_NAME must not be empty"
    exit 1
fi

root_dir=$(cd $(dirname $0)/.. && pwd)
ln -Fs ${root_dir}/models/research/slim/nets

TRAINER_PACKAGE_PATH=${root_dir}/trainer
MAIN_TRAINER_MODULE="trainer.task"
PACKAGE_STAGING_PATH="gs://${BUCKET_NAME}"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="shogi_$now"
JOB_DIR="gs://${BUCKET_NAME}/output/${JOB_NAME}"
REGION="asia-east1"

gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region $REGION \
    --config ${root_dir}/trainer/config.yaml \
    --runtime-version 1.10 \
    -- \
    --image_dir "gs://${BUCKET_NAME}/dataset" \
    --number_of_steps 100
