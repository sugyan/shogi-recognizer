#!/bin/bash

if [ -z "${BUCKET_NAME}" ]; then
    echo "BUCKET_NAME must not be empty"
    exit 1
fi

root_dir=$(cd $(dirname $0)/.. && pwd)
REGION="asia-east1"

if gsutil ls gs://${BUCKET_NAME} > /dev/null; then
    echo "bucket ${BUCKET_NAME} already exists"
else
    gsutil mb -l $REGION gs://${BUCKET_NAME}
fi

gsutil -m cp -r ${root_dir}/dataset gs://${BUCKET_NAME}
