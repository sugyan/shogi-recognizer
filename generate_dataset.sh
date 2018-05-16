#!/usr/bin/env bash

for downloader in scripts/download/*.py; do
    ${downloader}
done

# scripts/dataset/generate.py --fonts /System/Library/Fonts/ヒラギノ*.ttc
