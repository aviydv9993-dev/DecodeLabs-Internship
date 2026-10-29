"""
DecodeLabs | AI Internship | Project 3: AI Recommendation Logic
===================================================================
Tech Stack Recommender: content-based filtering with TF-IDF + cosine similarity.

Pipeline (IPO, 4-step ranking):
    1. INGESTION : capture >= 3 skills from the user (cold-start safe)
    2. SCORING   : cosine similarity of the user vector vs every job-role vector
    3. SORTING   : rank roles by score, descending
    4. FILTERING : keep only the Top-N (default 3)

TF-IDF and cosine similarity are implemented from scratch with NumPy so every
step of the maths is visible. A self-check against scikit-learn is included:
    python tech_stack_recommender.py --verify

Usage:
    python tech_stack_recommender.py                      # interactive
    python tech_stack_recommender.py -s python "cloud computing" automation
    python tech_stack_recommender.py -s python sql ml -n 5
"""

import argparse
import csv
import difflib
import math
import sys
from pathlib import Path

import numpy as np

DATA_FILE = Path(__file__).with_name("raw_skills.csv")
MIN_SKILLS = 3
TOP_N = 3

# Shared vocabulary: user wording must map onto dataset wording, otherwise the
# similarity maths silently fails ("Web Design" vs "Frontend Development").
ALIASES = {
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "machine learning",
    "cloud computing": "cloud",
    "amazon web services": "aws",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "node": "node.js",
    "nodejs": "node.js",
    "reactjs": "react",
    "cicd": "ci/cd",
    "ci cd": "ci/cd",
    "api": "apis",
    "rest api": "apis",
    "natural language processing": "nlp",
    "computer vision": "opencv",
    "dsa": "data structures",
    "algorithms": "data structures",
    "cpp": "c++",
    "postgres": "sql",
    "mysql": "sql",
    "data viz": "data visualization",
    "pen testing": "penetration testing",
    "infosec": "security",
    "cyber security": "security",
}


# --------------------------------------------------------------------------
# INPUT: dataset + vocabulary
# --------------------------------------------------------------------------
def normalize(skill: str) -> str:
    s = " ".join(skill.strip().lower().split())
    return ALIASES.get(s, s)


def load_roles(path: Path = DATA_FILE) -> dict:
    """Return {role: [skills]} from the CSV."""
    roles = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            roles[row["role"].strip()] = [
                normalize(s) for s in row["skills"].split(";") if s.strip()
            ]
    if not roles:
        raise ValueError(f"No roles found in {path}")
    return roles


def build_vocabulary(roles: dict) -> list:
    return sorted({s for skills in roles.values() for s in skills})


# --------------------------------------------------------------------------
# PROCESS: TF-IDF + cosine similarity (from scratch)
# --------------------------------------------------------------------------
def compute_idf(roles: dict, vocab: list) -> np.ndarray:
    """
    IDF = ln((N + 1) / (df + 1)) + 1
    Smoothed form of the slide's ln(N / df): the +1 terms stop a skill that
    appears in every role from getting weight 0 and avoid division by zero.
    """
    n_docs = len(roles)
    df = np.array([sum(term in skills for skills in roles.values())
                   for term in vocab], dtype=float)
    return np.log((n_docs + 1) / (df + 1)) + 1.0


def tfidf_vector(skills: list, vocab: list, idf: np.ndarray) -> np.ndarray:
    """TF = count of term / total terms in the document, then x IDF."""
    index = {t: i for i, t in enumerate(vocab)}
    vec = np.zeros(len(vocab))
    total = len(skills)
    for s in skills:
        if s in index:
            vec[index[s]] += 1
    return (vec / total if total else vec) * idf


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(theta) = (A . B) / (||A|| ||B||); returns 0.0 for a zero vector."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


# --------------------------------------------------------------------------
# 4-step ranking pipeline
# --------------------------------------------------------------------------
def ingest(raw_skills: list, vocab: list):
    """Step 1: clean input, split into known / unknown, enforce the minimum."""
    cleaned, seen = [], set()
    for s in raw_skills:
        n = normalize(s)
        if n and n not in seen:
            seen.add(n)
            cleaned.append(n)
    known = [s for s in cleaned if s in vocab]
    unknown = [s for s in cleaned if s not in vocab]
    return known, unknown


def score(user_skills: list, roles: dict, vocab: list, idf: np.ndarray) -> list:
    """Step 2: cosine score of the user profile against every role."""
    user_vec = tfidf_vector(user_skills, vocab, idf)
    results = []
    for role, skills in roles.items():
        sim = cosine_similarity(user_vec, tfidf_vector(skills, vocab, idf))
        results.append({
            "role": role,
            "score": sim,
            "matched": [s for s in skills if s in user_skills],
            "missing": [s for s in skills if s not in user_skills],
        })
    return results


def rank(results: list) -> list:
    """Step 3: sort by score (desc); role name breaks ties deterministically."""
    return sorted(results, key=lambda r: (-r["score"], r["role"]))


def top_n(results: list, n: int = TOP_N) -> list:
    """Step 4: truncate to avoid choice overload."""
    return results[:n]


def recommend(raw_skills: list, roles: dict, n: int = TOP_N) -> dict:
    vocab = build_vocabulary(roles)
    idf = compute_idf(roles, vocab)
    known, unknown = ingest(raw_skills, vocab)
    if len(known) < MIN_SKILLS:
        return {"error": True, "known": known, "unknown": unknown, "vocab": vocab}
    return {"error": False, "known": known, "unknown": unknown,
            "top": top_n(rank(score(known, roles, vocab, idf)), n)}


# --------------------------------------------------------------------------
# OUTPUT: display + CLI
# --------------------------------------------------------------------------
def suggest(skill: str, vocab: list) -> str:
    close = difflib.get_close_matches(skill, vocab, n=3, cutoff=0.6)
    return f"  - '{skill}' not recognised" + (
        f" (did you mean: {', '.join(close)}?)" if close else "")


def show(result: dict) -> None:
    print(f"\nSkills used : {', '.join(result['known'])}")
    if result["unknown"]:
        print(f"Ignored     : {', '.join(result['unknown'])}")
    print("\n" + "=" * 62)
    print(f" TOP {len(result['top'])} CAREER MATCHES")
    print("=" * 62)
    for i, r in enumerate(result["top"], 1):
        bar = "#" * round(r["score"] * 30)
        print(f"\n{i}. {r['role']}  -  {r['score'] * 100:.1f}% match")
        print(f"   [{bar:<30}]")
        print(f"   You have : {', '.join(r['matched']) or '-'}")
        print(f"   To learn : {', '.join(r['missing']) or '-'}")


def ask_interactively(vocab: list) -> list:
    """Cold-start bypass: onboarding survey that insists on enough data."""
    print("Tech Stack Recommender")
    print(f"Enter at least {MIN_SKILLS} skills, separated by commas.")
    print("Examples:", ", ".join(vocab[:8]), "...\n")
    while True:
        raw = input("Your skills > ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            sys.exit(0)
        skills = [s for s in raw.split(",") if s.strip()]
        known, unknown = ingest(skills, vocab)
        if len(known) >= MIN_SKILLS:
            return skills
        print(f"\nOnly {len(known)} recognised skill(s); need {MIN_SKILLS}.")
        for u in unknown:
            print(suggest(u, vocab))
        print("Try again (or type 'q' to quit).\n")


def verify() -> None:
    """Prove the from-scratch maths matches scikit-learn's implementation."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos

    roles = load_roles()
    vocab = build_vocabulary(roles)
    idf = compute_idf(roles, vocab)
    sample = ["python", "cloud", "automation"]

    mine = [cosine_similarity(tfidf_vector(sample, vocab, idf),
                              tfidf_vector(s, vocab, idf))
            for s in roles.values()]

    vec = TfidfVectorizer(analyzer=lambda doc: doc, vocabulary=vocab)
    docs = vec.fit_transform(list(roles.values()))
    ref = sk_cos(vec.transform([sample]), docs)[0]

    diff = float(np.max(np.abs(np.array(mine) - ref)))
    print(f"Max difference vs scikit-learn: {diff:.2e}")
    print("PASS" if diff < 1e-9 else "FAIL")


def main() -> None:
    p = argparse.ArgumentParser(description="Tech Stack Recommender")
    p.add_argument("-s", "--skills", nargs="+", help="skills (quote multi-word)")
    p.add_argument("-n", "--top", type=int, default=TOP_N, help="results to show")
    p.add_argument("--data", type=Path, default=DATA_FILE, help="skills CSV")
    p.add_argument("--verify", action="store_true", help="check vs scikit-learn")
    args = p.parse_args()

    if args.verify:
        verify()
        return

    roles = load_roles(args.data)
    vocab = build_vocabulary(roles)
    raw = args.skills or ask_interactively(vocab)
    result = recommend(raw, roles, args.top)

    if result["error"]:
        print(f"Need at least {MIN_SKILLS} recognised skills; "
              f"got {len(result['known'])}.")
        for u in result["unknown"]:
            print(suggest(u, vocab))
        sys.exit(1)
    show(result)


if __name__ == "__main__":
    main()
