#!/usr/bin/env bash
set -euo pipefail

python skipgram_homework.py --epochs 3
python submit.py --name "your-name" --model skipgram
