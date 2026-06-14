import argparse
from pathlib import Path

import requests

DEFAULT_URL_BROKER = "https://tues-ai.github.io/word2vec/server-url/current.txt"


def get_server_url(args):
    if args.server_url:
        return args.server_url.rstrip("/")

    response = requests.get(args.url_broker, timeout=10)
    response.raise_for_status()
    server_url = response.text.strip().rstrip("/")
    if not server_url:
        raise RuntimeError("Leaderboard server is sleeping. Try again when the tunnel is online.")
    return server_url


def print_top_rows(rows, limit=10):
    print("\nTop leaderboard rows:")
    for row in rows[:limit]:
        print(f"#{row['rank']:>2} {row['name']:<24} accuracy={row['accuracy'] * 100:6.2f}% loss={row['loss']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Submit a CBOW or Skip-gram Word2Vec checkpoint to the leaderboard.")
    parser.add_argument("--name", default="", help="leaderboard name; omit to get a score without saving")
    parser.add_argument("--model", choices=["cbow", "skipgram"], default="", help="leaderboard to submit to; inferred from checkpoint when omitted")
    parser.add_argument("--file", default="submission.pt", help="checkpoint file")
    parser.add_argument("--server-url", default="", help="direct server URL, useful for local testing")
    parser.add_argument("--url-broker", default=DEFAULT_URL_BROKER, help="URL that contains the current live server URL")
    parser.add_argument("--score-only", action="store_true", help="test score without saving to the leaderboard")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint file: {path}")

    server_url = get_server_url(args)
    route = "api/score" if args.score_only else "api/submit"

    form = {}
    if args.name:
        form["name"] = args.name
    if args.model:
        form["model_type"] = args.model

    with path.open("rb") as f:
        response = requests.post(
            f"{server_url}/{route}",
            data=form,
            files={"model_file": (path.name, f, "application/octet-stream")},
            timeout=300,
        )

    if not response.ok:
        print(response.text)
    response.raise_for_status()

    data = response.json()
    score = data["score"]
    print(f"Model: {score['model_type']}")
    print(f"Accuracy: {score['accuracy'] * 100:.2f}%")
    print(f"Loss: {score['loss']:.4f}")
    print(f"Perplexity: {score['perplexity']:.2f}")
    if not data.get("saved_to_leaderboard", True):
        print(data.get("message", "Not saved to leaderboard."))
    print_top_rows(data.get("leaderboard", []))


if __name__ == "__main__":
    main()
