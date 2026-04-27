#Fetches public ML Q&A from Stack Exchange (CrossValidated + Data Science)
#to expand the fine-tuning dataset well beyond the 40 hand-curated pairs.
#Output is JSON in the same {question, answer} shape as data/qa_dataset.json,
#so it plugs straight into the existing pipeline (or can be merged in).
#
#Usage:
#  python3 fetch_public_data.py --site stats --max 200
#  python3 fetch_public_data.py --site datascience --max 200 --merge
#
#Stack Exchange API: https://api.stackexchange.com/docs
#Anonymous quota is 300 req/day. We minimise calls by batching all answer
#lookups for up to 100 question ids in one request. Set STACKEXCHANGE_KEY
#in the environment for 10k/day if you need more headroom.
import os
import re
import json
import html
import time
import gzip
import argparse
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error  import HTTPError


DATA_DIR     = "data"
CURATED_FILE = os.path.join(DATA_DIR, "qa_dataset.json")
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
    s = html.unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def api_get(path, params, retries=3):
    key = os.environ.get("STACKEXCHANGE_KEY")
    if key:
        params = dict(params, key=key)
    url = f"{API}/{path}?{urlencode(params)}"
    req = Request(url, headers={
        "Accept-Encoding": "gzip",
        "User-Agent":      "ml-tutor-lora/0.1 (+https://github.com/Eric-Ristol/ml-tutor-lora)",
    })
    last_err = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=30) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                payload = json.loads(data)
            #SE uses a `backoff` field to ask clients to wait before next call.
            backoff = payload.get("backoff", 0)
            if backoff:
                print(f"    (SE backoff {backoff}s)")
                time.sleep(backoff)
            return payload
        except HTTPError as e:
            last_err = e
            if e.code == 429 or e.code >= 500:
                wait = 5 * (attempt + 1)
                print(f"    HTTP {e.code} — waiting {wait}s and retrying")
                time.sleep(wait)
                continue
            raise
    raise last_err


def fetch_top_questions(site, tag, pages=2, pagesize=50):
    questions = []
    for page in range(1, pages + 1):
        data = api_get("questions", {
            "site":     site,
            "tagged":   tag,
            "sort":     "votes",
            "order":    "desc",
            "pagesize": pagesize,
            "page":     page,
            "filter":   "withbody",
        })
        questions.extend(data.get("items", []))
        if not data.get("has_more"):
            break
        time.sleep(0.3)
    return questions


def fetch_answers_batch(site, qids):
    #SE allows up to 100 ids semicolon-joined. Returns the highest-voted
    #answer for each question.
    out = {}
    for i in range(0, len(qids), 100):
        chunk = qids[i:i + 100]
        ids   = ";".join(str(q) for q in chunk)
        data  = api_get(f"questions/{ids}/answers", {
            "site":     site,
            "sort":     "votes",
            "order":    "desc",
            "pagesize": 100,
            "filter":   "withbody",
        })
        for ans in data.get("items", []):
            qid = ans.get("question_id")
            #Keep only the first (highest-voted) answer per question.
            if qid not in out:
                out[qid] = ans
        time.sleep(0.3)
    return out


def fetch_pairs(site, tags=ML_TAGS, max_pairs=200, min_score=5,
                min_answer_chars=200, max_answer_chars=1500):
    candidates = []
    seen_qids  = set()

    for tag in tags:
        try:
            questions = fetch_top_questions(site, tag, pages=2, pagesize=50)
        except Exception as e:
            print(f"  ! tag={tag}: {e}")
            continue
        for q in questions:
            qid = q.get("question_id")
            if qid in seen_qids or q.get("score", 0) < min_score:
                continue
            seen_qids.add(qid)
            candidates.append((tag, q))

    print(f"Collected {len(candidates)} candidate questions; fetching answers...")
    qids_only = [q["question_id"] for _, q in candidates]
    try:
        answers = fetch_answers_batch(site, qids_only)
    except Exception as e:
        print(f"  ! answer batch failed: {e}")
        answers = {}

    pairs = []
    for tag, q in candidates:
        if len(pairs) >= max_pairs:
            break
        qid    = q["question_id"]
        answer = answers.get(qid)
        if not answer or answer.get("score", 0) < min_score:
            continue
        answer_text = html_to_text(answer.get("body", ""))
        if not (min_answer_chars <= len(answer_text) <= max_answer_chars):
            continue
        pairs.append({
            "question": html.unescape(q["title"]),
            "answer":   answer_text,
            "source":   f"{site}.stackexchange.com/q/{qid}",
            "tag":      tag,
        })
        print(f"  + [{site}/{tag}] {q['title'][:70]}")
    return pairs


def load_curated():
    #Read the curated dataset from JSON without importing data.py
    #(data.py imports the heavy `datasets` library).
    if not os.path.exists(CURATED_FILE):
        raise FileNotFoundError(
            f"{CURATED_FILE} not found. Run  python3 data.py  first to "
            "generate the curated JSON, then re-run this script."
        )
    with open(CURATED_FILE, encoding="utf-8") as f:
        return json.load(f)


def merge_with_curated(public_pairs):
    curated_raw = load_curated()
    curated     = [{"question": p["question"], "answer": p["answer"]} for p in curated_raw]
    public_min  = [{"question": p["question"], "answer": p["answer"]} for p in public_pairs]
    merged      = curated + public_min
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MERGED_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nMerged {len(curated)} curated + {len(public_min)} public "
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

    if args.merge and pairs:
        merge_with_curated(pairs)
    elif args.merge:
        print("Skipping merge (no public pairs collected).")


if __name__ == "__main__":
    main()
