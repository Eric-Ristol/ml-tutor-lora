#Fetches public ML Q&A from Stack Exchange (CrossValidated + Data Science)
#to expand the fine-tuning dataset well beyond the 40 hand-curated pairs.
#Output is JSON in the same {question, answer} shape as data/qa_dataset.json,
#so it plugs straight into the existing pipeline (or can be merged in).
#
#Usage:
#  python fetch_public_data.py --site stats --max 200
#  python fetch_public_data.py --site datascience --max 200 --merge
#
#Stack Exchange API: https://api.stackexchange.com/docs
#Rate limit: 300 req/day anon; set STACKEXCHANGE_KEY env var for 10k/day.
import os
import re
import json
import html
import time
import gzip
import argparse
from urllib.parse import urlencode
from urllib.request import urlopen, Request


DATA_DIR     = "data"
OUTPUT_FILE  = os.path.join(DATA_DIR, "qa_dataset_public.json")
MERGED_FILE  = os.path.join(DATA_DIR, "qa_dataset_merged.json")
API          = "https://api.stackexchange.com/2.3"

#Tags that map well to the ML-tutor scope.
ML_TAGS = [
    "machine-learning",
    "neural-networks",
    "deep-learning",
    "regression",
    "classification",
    "regularization",
    "overfitting",
    "cross-validation",
    "gradient-descent",
    "feature-selection",
]

_TAG_RE        = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def html_to_text(s):
    #Strip HTML tags and collapse whitespace so answers are clean plain text.
    s = html.unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def api_get(path, params):
    key = os.environ.get("STACKEXCHANGE_KEY")
    if key:
        params = dict(params, key=key)
    url = f"{API}/{path}?{urlencode(params)}"
    req = Request(url, headers={"Accept-Encoding": "gzip"})
    with urlopen(req, timeout=30) as r:
        data = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return json.loads(data)


def fetch_top_questions(site, tag, pages=2, pagesize=50):
    questions = []
    for page in range(1, pages + 1):
        params = {
            "site":     site,
            "tagged":   tag,
            "sort":     "votes",
            "order":    "desc",
            "pagesize": pagesize,
            "page":     page,
            "filter":   "withbody",
        }
        data = api_get("questions", params)
        questions.extend(data.get("items", []))
        if not data.get("has_more"):
            break
        time.sleep(0.2)
    return questions


def fetch_top_answer(site, question_id):
    params = {
        "site":     site,
        "sort":     "votes",
        "order":    "desc",
        "pagesize": 1,
        "filter":   "withbody",
    }
    data = api_get(f"questions/{question_id}/answers", params)
    items = data.get("items", [])
    return items[0] if items else None


def fetch_pairs(site, tags=ML_TAGS, max_pairs=200, min_score=5,
                min_answer_chars=200, max_answer_chars=1500):
    pairs = []
    seen  = set()
    for tag in tags:
        if len(pairs) >= max_pairs:
            break
        try:
            questions = fetch_top_questions(site, tag, pages=2, pagesize=50)
        except Exception as e:
            print(f"  ! tag={tag}: {e}")
            continue

        for q in questions:
            if len(pairs) >= max_pairs:
                break
            qid = q.get("question_id")
            if qid in seen or q.get("score", 0) < min_score:
                continue

            answer = fetch_top_answer(site, qid)
            time.sleep(0.2)
            if not answer or answer.get("score", 0) < min_score:
                continue

            answer_text = html_to_text(answer.get("body", ""))
            if not (min_answer_chars <= len(answer_text) <= max_answer_chars):
                continue

            seen.add(qid)
            pairs.append({
                "question": html.unescape(q["title"]),
                "answer":   answer_text,
                "source":   f"{site}.stackexchange.com/q/{qid}",
                "tag":      tag,
            })
            print(f"  + [{site}/{tag}] {q['title'][:70]}")

    return pairs


def merge_with_curated(public_pairs):
    #Combine the curated 40 pairs with public-dataset pairs.
    #Curated pairs come first so they keep their formatting weight in training.
    from data import QA_PAIRS
    curated = [{"question": p["question"], "answer": p["answer"]} for p in QA_PAIRS]
    merged  = curated + public_pairs
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MERGED_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nMerged {len(curated)} curated + {len(public_pairs)} public "
          f"= {len(merged)} -> {MERGED_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="stats",
                        choices=["stats", "datascience"],
                        help="Stack Exchange site (stats = CrossValidated)")
    parser.add_argument("--max", type=int, default=200,
                        help="Max pairs to collect")
    parser.add_argument("--min-score", type=int, default=5,
                        help="Minimum score for both question and answer")
    parser.add_argument("--merge", action="store_true",
                        help="Also write merged dataset (curated + public)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Fetching from {args.site}.stackexchange.com "
          f"(max={args.max}, min_score={args.min_score})")
    pairs = fetch_pairs(args.site, max_pairs=args.max, min_score=args.min_score)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(pairs)} pairs -> {OUTPUT_FILE}")

    if args.merge:
        merge_with_curated(pairs)


if __name__ == "__main__":
    main()
