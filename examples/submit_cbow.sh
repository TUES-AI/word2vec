#!/usr/bin/env bash
set -euo pipefail

python cbow_word2vec.py --epochs 3
python submit.py --name "your-name" --model cbow --file cbow_submission.pt
