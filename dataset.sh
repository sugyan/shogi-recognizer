#!/usr/bin/env bash

python scripts/muchonovski/download.py
python scripts/muchonovski/dataset.py

python scripts/irasutoya/download.py
python scripts/irasutoya/dataset.py

python scripts/shineleckoma/download.py
python scripts/shineleckoma/dataset.py

python scripts/character/dataset.py --fonts 
