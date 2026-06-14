from pathlib import Path
import random
import sys

TEXT_FILE = "text.txt"
N = 2  # try 2 or 3
SEED = "the cat"
MAX_WORDS = 20
RANDOM_SEED = 1

START = "<START>"
END = "<END>"


def read_lines():
    path = Path(__file__).with_name(TEXT_FILE)
    return path.read_text(encoding="utf-8").lower().splitlines()


def train_ngram(n):
    counts = {}

    for line in read_lines():
        words = line.split()
        padding = [START] * (n - 1)
        words = padding + words + [END]

        for i in range(len(words) - n + 1):
            ngram_words = words[i : i + n]
            ngram = tuple(ngram_words)

            if ngram not in counts:
                counts[ngram] = 0

            counts[ngram] = counts[ngram] + 1

    return counts


def next_word(counts, context):
    possible_words = []
    possible_counts = []

    for ngram, count in counts.items():
        ngram_context = ngram[:-1]
        ngram_next_word = ngram[-1]

        if ngram_context == context:
            possible_words.append(ngram_next_word)
            possible_counts.append(count)

    if not possible_words:
        return None

    picked_words = random.choices(
        possible_words,
        weights=possible_counts,
        k=1,
    )
    return picked_words[0]


def generate_text(counts, seed, n):
    words = seed.lower().split()

    for _ in range(MAX_WORDS):
        padding = [START] * (n - 1)
        context_words = padding + words
        context_words = context_words[-(n - 1) :]
        context = tuple(context_words)

        word = next_word(counts, context)

        if word is None or word == END:
            break

        words.append(word)

    return " ".join(words)


def read_settings():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N
    seed = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else SEED
    return n, seed


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    n, seed = read_settings()
    counts = train_ngram(n)
    print(generate_text(counts, seed, n))
