#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Discord term-level analysis:
1) Unigrams (single corpus): term frequency (TF) + document frequency (DF)
2) Bigrams  (single corpus): TF + DF
3) Bigrams  (two corpora: Papyrology vs General): Mutual Information (MI)

Enhancements:
- Stemming (Snowball/English) toggleable via --no_stem.
- Comprehensive stopwords: scikit-learn ENGLISH_STOP_WORDS + extended contractions/fillers + user extras.
- Custom analyzers:
    * Unigrams: tokenized, stopworded, (optional) stemmed tokens.
    * Bigrams: adjacent-token pairs emitted as "a b" strings (so CountVectorizer truly gets bigrams).
- Removed over-strict bigram post-filter (the analyzer already ensures clean tokens).

Usage:
  python discord_term_analysis.py --input_dir ./json_dir --output_dir ./outputs
  # Optional:
  --extra_stopwords "ok,yeah,thing,things"   # comma-separated
  --no_stem                                   # to disable stemming
  --top_k 200 --min_doc_len 10
  --papyrology_pattern papyrolog --general_pattern general

Outputs (CSV):
  - unigrams_single_corpus.csv  (term, tf, df)
  - bigrams_single_corpus.csv   (term, tf, df)     # term holds the bigram string "a b"
  - mi_bigrams_general_vs_papyrology.csv (bigram, n11,n01,n10,n00, p_papy, p_gen, mi_bits)
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Stemming (no external downloads required for Snowball)
try:
    from nltk.stem import SnowballStemmer
    STEMMER_AVAILABLE = True
except Exception:
    STEMMER_AVAILABLE = False

# -----------------------
# Utilities: loading JSON
# -----------------------

SYSTEM_MSG_RE = re.compile(r"^\s*(Joined the server\.|Started a thread\.)\s*$", re.I)
URL_ONLY_RE   = re.compile(r"^\s*(https?://\S+\s*)+$", re.I)
URL_INLINE_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def load_discord_dir(input_dir: Path) -> pd.DataFrame:
    """Load Discord JSON exports into a message-level DataFrame."""
    rows = []
    for p in input_dir.rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        guild   = data.get("guild", {}) or {}
        channel = data.get("channel", {}) or {}
        messages = data.get("messages", []) or []
        for m in messages:
            txt = m.get("content")
            rows.append({
                "guild_id":     guild.get("id"),
                "guild_name":   guild.get("name"),
                "channel_id":   channel.get("id"),
                "channel_name": channel.get("name"),
                "category":     channel.get("category"),
                "message_id":   m.get("id"),
                "timestamp":    m.get("timestamp"),
                "text":         "" if txt is None else str(txt)
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["text"] = df["text"].astype(str)
    return df

# -----------------------
# Cleaning / Prep
# -----------------------

def clean_messages(df: pd.DataFrame, min_doc_len: int = 10) -> pd.DataFrame:
    """Remove blank/system/url-only; strip whitespace; keep messages >= min_doc_len chars; drop inline links."""
    if df.empty:
        return df
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    # remove empty
    df = df[df["text"] != ""]
    # remove boilerplate
    df = df[~df["text"].str.fullmatch(SYSTEM_MSG_RE, na=False)]
    # remove url-only rows
    df = df[~df["text"].str.fullmatch(URL_ONLY_RE, na=False)]
    # strip inline links
    df["text"] = df["text"].str.replace(URL_INLINE_RE, "", regex=True).str.strip()
    # normalize curly apostrophes → straight
    df["text"] = df["text"].str.replace("’", "'", regex=False)
    # enforce minimum length
    df = df[df["text"].str.len() >= int(min_doc_len)]
    return df

# -----------------------
# Stopwords & Analyzer
# -----------------------

# Extra contractions/fillers beyond scikit-learn’s list
DEFAULT_EXTRA_STOPS = {
    # frequent fillers
    "like", "also", "really", "just", "okay", "ok", "yeah", "yep", "nope", "kinda", "sorta",
    # common contractions (normalize to lower-case)
    "i'm", "im", "you're", "youre", "he's", "she's", "it's", "its", "we're", "weve", "we've",
    "they're", "theyre", "that's", "thats", "there's", "theres", "what's", "whats", "who's",
    "dont", "don't", "doesnt", "doesn't", "isnt", "isn't", "wasnt", "wasn't", "shouldn't",
    "shouldnt", "couldn't", "couldnt", "wouldn't", "wouldnt", "ain't", "aint", "won't", "wont",
    "can't", "cant", "haven't", "havent", "hasn't", "hasnt", "hadn't", "hadnt", "i'll", "ill",
    "you'll", "youll", "we'll", "well", "they'll", "theyll", "i'd", "id", "you'd", "youd",
    "we'd", "wed", "they'd", "theyd", "ive", "i've",
    # optional modals (remove if you prefer)
    "could", "would", "should",
}

def build_stopwords(extra: Optional[str]) -> Set[str]:
    """
    Build a comprehensive stopword set:
      - scikit-learn ENGLISH_STOP_WORDS
      - DEFAULT_EXTRA_STOPS
      - user provided via --extra_stopwords (comma-separated)
    """
    stops = set(ENGLISH_STOP_WORDS) | set(DEFAULT_EXTRA_STOPS)
    if extra:
        user_extra = {w.strip().lower() for w in extra.split(",") if w.strip()}
        stops |= user_extra
    return stops

def make_analyzer(ngram_mode: str,
                  do_stem: bool,
                  stopwords: Set[str]) -> Callable[[str], List[str]]:
    """
    Return a callable analyzer that scikit-learn will call.
    Important: When analyzer is callable, CountVectorizer ignores ngram_range,
    so this function must emit *exactly* the tokens/grams you want.
    - 'uni': returns unigrams (tokens)
    - 'bi' : returns bigram strings "a b" from adjacent tokens
    """
    if do_stem and STEMMER_AVAILABLE:
        stemmer = SnowballStemmer("english")
    else:
        stemmer = None

    # one base tokenizer for both modes
    base_token_re = re.compile(r"(?u)\b[\w']+\b")

    def base_tokens(text: str) -> List[str]:
        toks = base_token_re.findall(text.lower())
        # drop 1-char tokens
        toks = [t for t in toks if len(t) > 1]
        # stopword filter
        toks = [t for t in toks if t not in stopwords]
        # stemming
        if stemmer is not None:
            toks = [stemmer.stem(t) for t in toks]
        return toks

    if ngram_mode == "uni":
        return base_tokens

    # ngram_mode == "bi": return "a b" strings
    def bigram_analyzer(doc: str) -> List[str]:
        toks = base_tokens(doc)
        return [f"{a} {b}" for a, b in zip(toks, toks[1:])]

    return bigram_analyzer

# -----------------------
# Vectorization helpers
# -----------------------

def vectorize_counts(docs: List[str],
                     analyzer: Callable[[str], List[str]],
                     min_df: int = 1) -> Tuple[CountVectorizer, np.ndarray]:
    """
    Fit CountVectorizer with a *callable analyzer* and return (vectorizer, X).
    NOTE: With a callable analyzer, token_pattern/stop_words/ngram_range are ignored.
    """
    vect = CountVectorizer(
        lowercase=True,
        min_df=min_df,
        analyzer=analyzer
    )
    X = vect.fit_transform(docs)
    return vect, X

def term_stats_from_counts(X, feature_names: List[str]) -> pd.DataFrame:
    """Return DataFrame with columns: term/bigram, tf, df."""
    tf = np.asarray(X.sum(axis=0)).ravel()
    dfreq = np.asarray((X > 0).sum(axis=0)).ravel()
    out = pd.DataFrame({
        "term": feature_names,
        "tf": tf.astype(int),
        "df": dfreq.astype(int)
    }).sort_values(["tf","df","term"], ascending=[False, False, True])
    return out

# -----------------------
# Mutual Information (two corpora)
# -----------------------

def mutual_information_binary(bigram_presence: np.ndarray, labels: np.ndarray, smoothing: float = 0.5):
    """
    Compute mutual information I(X;Y) for each bigram presence X (binary) vs Y in {0,1}.
    Returns:
      mi_bits: shape [N_terms]
      counts: (n11, n01, n10, n00)
    """
    N, M = bigram_presence.shape
    y = labels.astype(int)

    n11 = (bigram_presence[y == 1].sum(axis=0)).A1 if hasattr(bigram_presence, "A1") else np.asarray(bigram_presence[y == 1].sum(axis=0)).ravel()
    n01 = (bigram_presence[y == 0].sum(axis=0)).A1 if hasattr(bigram_presence, "A1") else np.asarray(bigram_presence[y == 0].sum(axis=0)).ravel()

    n1_ = (y == 1).sum()
    n0_ = (y == 0).sum()

    n10 = n1_ - n11
    n00 = n0_ - n01

    # smoothing
    n11s = n11 + smoothing
    n01s = n01 + smoothing
    n10s = n10 + smoothing
    n00s = n00 + smoothing

    Ns = n11s + n01s + n10s + n00s

    # probabilities
    p11 = n11s / Ns
    p01 = n01s / Ns
    p10 = n10s / Ns
    p00 = n00s / Ns

    px1 = (n11s + n01s) / Ns
    px0 = (n10s + n00s) / Ns
    py1 = (n11s + n10s) / Ns
    py0 = (n01s + n00s) / Ns

    def safe_term(pxy, px, py):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = pxy / (px * py)
            term = np.where((pxy > 0) & (px > 0) & (py > 0), pxy * np.log2(ratio), 0.0)
        return term

    mi = (
        safe_term(p11, px1, py1) +
        safe_term(p10, px0, py1) +
        safe_term(p01, px1, py0) +
        safe_term(p00, px0, py0)
    )

    return mi, (n11, n01, n10, n00)

# -----------------------
# Corpus labeling for MI
# -----------------------

def label_corpora_for_mi(df: pd.DataFrame, pap_pat: str, gen_pat: str) -> pd.Series:
    """
    Return Series of labels for MI:
      1 = Papyrology, 0 = General, NaN = neither (excluded from MI step).
    Matches if channel_name OR category matches the regex (case-insensitive).
    """
    papy = df["channel_name"].fillna("").str.contains(pap_pat, case=False, regex=True) \
         | df["category"].fillna("").str.contains(pap_pat, case=False, regex=True)
    gen  = df["channel_name"].fillna("").str.contains(gen_pat,  case=False, regex=True) \
         | df["category"].fillna("").str.contains(gen_pat,  case=False, regex=True)

    labels = pd.Series(np.nan, index=df.index, dtype="float")
    labels[papy] = 1.0
    labels[gen & ~papy] = 0.0  # prefer papyrology if both match (rare)
    return labels

# -----------------------
# Main
# -----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./outputs")
    ap.add_argument("--min_doc_len", type=int, default=10, help="min characters per message to keep")
    ap.add_argument("--top_k", type=int, default=200, help="top rows to keep in outputs (use -1 for all)")
    ap.add_argument("--papyrology_pattern", type=str, default=r"papyrolog", help="regex to detect Papyrology channels/categories")
    ap.add_argument("--general_pattern", type=str, default=r"general", help="regex to detect General channels/categories")

    # New options for stemming/stopwords
    ap.add_argument("--no_stem", action="store_true",
                    help="Disable stemming (default is ON if nltk SnowballStemmer available).")
    ap.add_argument("--extra_stopwords", type=str, default="",
                    help="Comma-separated extra stopwords to remove in addition to defaults.")
    return ap.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading & cleaning messages…")
    df = load_discord_dir(input_dir)
    if df.empty:
        print("No messages found. Exiting.")
        return
    df = clean_messages(df, min_doc_len=args.min_doc_len)
    if df.empty:
        print("All messages filtered out. Exiting.")
        return

    docs = df["text"].tolist()
    print(f"Kept {len(docs):,} documents.")

    # Build stopword set & analyzers
    stops = build_stopwords(args.extra_stopwords)
    do_stem = (not args.no_stem) and STEMMER_AVAILABLE
    if (not STEMMER_AVAILABLE) and (not args.no_stem):
        print("Warning: NLTK SnowballStemmer not available; proceeding without stemming.")
    uni_analyzer = make_analyzer(ngram_mode="uni", do_stem=do_stem, stopwords=stops)
    bi_analyzer  = make_analyzer(ngram_mode="bi",  do_stem=do_stem, stopwords=stops)

    # ------------------ Single-corpus: UNIGRAMS ------------------
    print("[2/4] Unigram term-level analysis (single corpus)…")
    vec_uni, X_uni = vectorize_counts(docs, analyzer=uni_analyzer, min_df=1)
    uni_stats = term_stats_from_counts(X_uni, vec_uni.get_feature_names_out())
    uni_stats_out = uni_stats.head(args.top_k) if args.top_k > 0 else uni_stats
    uni_stats_out.to_csv(outdir / "unigrams_single_corpus.csv", index=False)

    # ------------------ Single-corpus: BIGRAMS -------------------
    print("[3/4] Bigram term-level analysis (single corpus)…")
    vec_bi_all, X_bi_all = vectorize_counts(docs, analyzer=bi_analyzer, min_df=2)  # keep min_df=2 to prune ultra-rare
    bi_stats_all = term_stats_from_counts(X_bi_all, vec_bi_all.get_feature_names_out())
    # No post-filter; analyzer already ensures clean grams
    bi_stats_all_out = bi_stats_all.head(args.top_k) if args.top_k > 0 else bi_stats_all
    bi_stats_all_out.to_csv(outdir / "bigrams_single_corpus.csv", index=False)

    # Diagnostics for bigrams
    print("Sample bigrams:", bi_stats_all["term"].head(10).tolist())
    print("Bigrams kept count:", len(bi_stats_all))

    # --------------- Two corpora: BIGRAM MI (Papy vs General) ---------------
    print("[4/4] Bigram MI (Papyrology vs General)…")
    labels = label_corpora_for_mi(df, args.papyrology_pattern, args.general_pattern)
    mi_mask = labels.notna()
    df_mi = df.loc[mi_mask].copy()
    y = labels.loc[mi_mask].astype(int).to_numpy()

    if df_mi.empty or y.size == 0 or len(np.unique(y)) < 2:
        print("Not enough labeled documents across both corpora for MI. Skipping MI step.")
        terms = []
    else:
        print(f"Label counts -> Papyrology: {(y==1).sum()}, General: {(y==0).sum()}")
        docs_mi = df_mi["text"].tolist()
        vec_bi_mi, X_bi_mi_counts = vectorize_counts(docs_mi, analyzer=bi_analyzer, min_df=2)
        # presence
        X_bi_mi_bin = (X_bi_mi_counts > 0).astype(int)

        # compute MI
        mi_bits, counts = mutual_information_binary(X_bi_mi_bin, y, smoothing=0.5)
        n11, n01, n10, n00 = counts

        terms = vec_bi_mi.get_feature_names_out()

        N = X_bi_mi_bin.shape[0]
        p_papy = (y == 1).sum() / N
        p_gen  = (y == 0).sum() / N

        mi_df = pd.DataFrame({
            "bigram": terms,
            "n11_papy_and_term": n11.astype(int),
            "n01_gen_and_term":  n01.astype(int),
            "n10_papy_no_term":  n10.astype(int),
            "n00_gen_no_term":   n00.astype(int),
            "p_papy_docs":       np.round(p_papy, 6),
            "p_gen_docs":        np.round(p_gen, 6),
            "mi_bits":           mi_bits
        }).sort_values(["mi_bits","n11_papy_and_term","n01_gen_and_term"], ascending=[False, False, False])

        mi_out = mi_df.head(args.top_k) if args.top_k > 0 else mi_df
        mi_out.to_csv(outdir / "mi_bigrams_general_vs_papyrology.csv", index=False)

    # Small summary
    print("\nSummary")
    print("-------")
    print(f"Documents total:      {len(docs):,}")
    print(f"Unigrams vocab size:  {X_uni.shape[1]:,}")
    print(f"Bigrams  vocab size:  {X_bi_all.shape[1]:,} (pre-sort)")
    print(f"Bigrams  kept:        {len(bi_stats_all):,}")
    if not df_mi.empty and y.size > 0 and len(np.unique(y)) >= 2 and len(terms) > 0:
        print(f"MI subset docs:       {X_bi_mi_bin.shape[0]:,} (Papy={int((y==1).sum())}, General={int((y==0).sum())})")
        print(f"MI bigrams analyzed:  {len(terms):,}")
    else:
        print("MI: skipped or no terms after vectorization.")
    print(f"Top-k saved:          {args.top_k if args.top_k>0 else 'ALL'}")
    print(f"Stemming:             {'ON' if do_stem else 'OFF'}")
    print(f"Extra stopwords:      {args.extra_stopwords if args.extra_stopwords else '(none)'}")
    print(f"Outputs in:           {outdir.resolve()}")

if __name__ == "__main__":
    main()
