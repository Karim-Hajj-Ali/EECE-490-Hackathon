"""
Utility functions:
- load_data(path)
- compute_priority(df)
- top_issues(df)
"""
import os
import requests
import re
from typing import List, Dict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Simple keyword weights for heuristics (extend as needed)
KEYWORD_WEIGHTS = {
    'urgent': 3,
    'asap': 3,
    'immediately': 3,
    'critical': 4,
    'critical.': 4,
    'down': 4,
    'outage': 4,
    'payment': 2,
    'billing': 2,
    'error': 2,
    'fail': 3,
    'failed': 3,
    'unable': 2,
    'lost': 4,
    'data loss': 5,
    'lost data': 5,
    'security': 4,
    'breach': 5,
}




def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    colmap = {
        "Ticket ID": "ticket_id",
        "Ticket Subject": "subject",
        "Ticket Description": "description",
        "Ticket Priority": "priority",
    }
    for k, v in colmap.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    if "ticket_id" not in df.columns:
        df["ticket_id"] = df.index.astype(str)
    return df


def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = re.sub(r'[\r\n]+', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_priority(df):
    """
    Compute a more fine-grained priority score (0-10) using heuristics.
    - Uses keywords, text length, and all-caps emphasis.
    - Maps numeric score to labels:
        high: 7-10
        medium: 4-6
        low: 0-3
    """
    out = df.copy()

    # Compute heuristic score
    def heur_score(row):
        text = ' '.join(str(row.get(c, '') or '') for c in ('subject', 'description'))
        clean = _clean_text(text)
        score = 0.0

        # keyword matches (weighted)
        for kw, w in KEYWORD_WEIGHTS.items():
            if kw in clean:
                score += w

        # text length boost
        l = len(clean.split())
        if l > 50:
            score += 2.0
        elif l > 20:
            score += 1.0

        # all-caps words in original text
        raw = (row.get('subject') or '') + ' ' + (row.get('description') or '')
        if re.search(r'\b[A-Z]{3,}\b', str(raw)):
            score += 1.5

        return score

    # compute raw heuristic for all rows
    raw_scores = out.apply(heur_score, axis=1)

    # normalize to 0-10
    max_score = raw_scores.max() if raw_scores.max() > 0 else 1
    out['priority_score'] = raw_scores / max_score * 10

    # map to labels
    def label_from_score(s):
        if s >= 7:
            return 'high'
        elif s >= 4:
            return 'medium'
        else:
            return 'low'

    out['priority_label'] = out['priority_score'].apply(label_from_score)

    # sort by score descending
    out = out.sort_values('priority_score', ascending=False).reset_index(drop=True)
    return out


    # compute heuristic score for rows missing priority_score
    def heur_score(row):
        text = ' '.join(str(row.get(c, '') or '') for c in ('subject', 'description'))
        clean = _clean_text(text)
        score = 0.0
        # keyword matches
        for kw, w in KEYWORD_WEIGHTS.items():
            if kw in clean:
                score += w
        # text length (longer may mean more details -> possibly complex/urgent) - small effect
        l = len(clean.split())
        if l > 50:
            score += 2.0
        elif l > 20:
            score += 1.0
        # presence of all-caps words in original text (often emphasis)
        raw = (row.get('subject') or '') + ' ' + (row.get('description') or '')
        if re.search(r'\b[A-Z]{3,}\b', str(raw)):
            score += 1.5
        return score

    # fill missing numeric scores with heuristic
    out['priority_score'] = out['priority_score'].apply(lambda v: v if (v is not None and not pd.isna(v)) else None)
    out['priority_score'] = out.apply(lambda r: r['priority_score'] if r['priority_score'] is not None else heur_score(r), axis=1)

    # Map numeric score to label
    def label_from_score(s):
        if s >= 8:
            return 'high'
        if s >= 4:
            return 'medium'
        return 'low'

    out['priority_label'] = out['priority_score'].apply(label_from_score)
    out = out.sort_values('priority_score', ascending=False).reset_index(drop=True)
    return out

def top_issues(df, top_n=20):
    """
    Return top_n bigram terms across subject+description,
    weighted by ticket priority score so urgent tickets count more.
    """
    # Combine subject + description
    texts = (df.get('subject', pd.Series('')) + ' ' + df.get('description', pd.Series(''))).fillna('').astype(str).tolist()
    priorities = df.get('priority_score', pd.Series(1)).tolist()  # fallback to 1 if missing

    vect = CountVectorizer(stop_words='english', ngram_range=(2,2), min_df=2)
    X = vect.fit_transform(texts)  # shape (n_tickets, n_terms)
    terms = vect.get_feature_names_out()

    # Multiply each term occurrence by ticket priority score
    weighted_counts = (X.multiply(priorities)).sum(axis=0).A1  # sums weighted by priority
    counts_sorted = sorted(zip(terms, weighted_counts), key=lambda x: x[1], reverse=True)

    # Return top N bigrams
    return [(t, int(c)) for t,c in counts_sorted[:top_n]]

