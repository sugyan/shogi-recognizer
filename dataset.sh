#!/usr/bin/env bash

SOURCES='muchonovski irasutoya shineleckoma sozai'

for SOURCE in ${SOURCES}; do
    scripts/${SOURCE}/download.py
done

for SOURCE in ${SOURCES}; do
    scripts/${SOURCE}/dataset.py
done
scripts/character/dataset.py --fonts /System/Library/Fonts/ヒラギノ*.ttc
