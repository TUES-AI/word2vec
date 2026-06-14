# [Word2Vec Challenge](https://tues-ai.github.io/word2vec/)

Student repo for CBOW and Skip-gram Word2Vec with a GitHub Pages leaderboard.

## What to implement

Main homework: complete the TODOs in [skipgram_homework.py](skipgram_homework.py).

- `SkipGramDataset`: build `(center_word_id, context_word_id)` pairs.
- `SkipGramWord2Vec`: define `self.embedding` and `self.output`.
- `forward`: turn center ids into vocabulary logits.

CBOW is already implemented in [cbow_word2vec.py](cbow_word2vec.py) and can also be submitted.

## Objectives

Skip-gram minimizes cross-entropy for predicting a nearby context word from the center word.

CBOW minimizes cross-entropy for predicting the center word from its surrounding context words.

Leaderboard ranking uses held-out accuracy; lower loss breaks ties.

## Train

Skip-gram:

```bash
python skipgram_homework.py --epochs 3
```

CBOW:

```bash
python cbow_word2vec.py --epochs 3
```

Keep the starter defaults for leaderboard-compatible checkpoints: vocab size `5000`, window size `2`, max tokens `200000`, validation fraction `0.2`.

## Submit

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Score without saving by omitting `--name`:

```bash
python submit.py --model skipgram
python submit.py --model cbow --file cbow_submission.pt
```

You can also force test-only mode with a name:

```bash
python submit.py --name "your-name" --model skipgram --score-only
python submit.py --name "your-name" --model cbow --file cbow_submission.pt --score-only
```

Publish by adding a name:

```bash
python submit.py --name "your-name" --model skipgram
python submit.py --name "your-name" --model cbow --file cbow_submission.pt
```

Leaderboard page: https://tues-ai.github.io/word2vec/

## Repo layout

- [skipgram_homework.py](skipgram_homework.py) — student starter.
- [cbow_word2vec.py](cbow_word2vec.py) — simple CBOW implementation.
- [submit.py](submit.py) — terminal submission client.
- [index.html](index.html), [server-url/](server-url/) — GitHub Pages leaderboard and live backend URL broker.
- [examples/](examples/) — copy-paste command examples.

