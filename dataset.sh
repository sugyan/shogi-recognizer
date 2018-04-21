#!/usr/bin/env bash

SOURCES='muchonovski irasutoya shineleckoma sozai'

# for SOURCE in ${SOURCES}; do
#     python scripts/${SOURCE}/download.sh
# done

for SOURCE in ${SOURCES}; do
    python ./scripts/${SOURCE}/dataset.py
done
python scripts/character/dataset.py --fonts /System/Library/Fonts/ヒラギノ*.ttc
