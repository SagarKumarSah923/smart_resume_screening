"""
model.py
--------
NLP-based resume–job description similarity scoring using TF-IDF + cosine similarity.
Also provides keyword gap analysis for actionable feedback.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import clean_text

# ---------------------------------------------------------------------------
# Core Similarity
# ---------------------------------------------------------------------------

def calculate_similarity(resume_text: str, job_desc: str) -> float:
    """
    Compute cosine similarity between resume and job description using TF-IDF.

    Args:
        resume_text: Raw or cleaned resume text.
        job_desc:    Raw or cleaned job description text.

    Returns:
        Similarity score as a percentage (0.0 – 100.0).
    """
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_desc)

    if not cleaned_resume or not cleaned_jd:
        return 0.0

    corpus = [cleaned_resume, cleaned_jd]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams for richer matching
        min_df=1,
    )
    vectors = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(float(similarity[0][0]) * 100, 2)


# ---------------------------------------------------------------------------
# Keyword Gap Analysis
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Return a set of meaningful words (length > 2) from text."""
    text = clean_text(text)
    words = re.findall(r"\b[a-z]{3,}\b", text)
    # Remove generic stopwords manually (TfidfVectorizer's list is available
    # but we want a lightweight set here)
    generic = {
        "the", "and", "for", "are", "was", "were", "this", "that", "with",
        "have", "has", "had", "will", "our", "your", "their", "from", "you",
        "use", "used", "using", "can", "also", "able", "must", "may", "all",
        "any", "not", "but", "its", "per", "via", "etc", "both", "each",
        "more", "than", "other", "well", "new", "get", "set", "work",
    }
    return set(w for w in words if w not in generic)


def get_keyword_analysis(resume_text: str, job_desc: str) -> dict:
    """
    Identify matched and missing keywords between resume and job description.

    Args:
        resume_text: Raw resume text.
        job_desc:    Raw job description text.

    Returns:
        Dict with keys:
            - matched  : list of keywords present in both
            - missing  : list of JD keywords absent from resume
            - coverage : percentage of JD keywords covered
    """
    resume_words = _tokenize(resume_text)
    jd_words = _tokenize(job_desc)

    matched = sorted(resume_words & jd_words)
    missing = sorted(jd_words - resume_words)
    coverage = round(len(matched) / max(len(jd_words), 1) * 100, 1)

    return {
        "matched": matched,
        "missing": missing,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Candidate Classification
# ---------------------------------------------------------------------------

def classify_candidate(score: float) -> dict:
    """
    Return a risk tier label, colour, and recommendation based on match score.

    Args:
        score: Similarity percentage (0–100).

    Returns:
        Dict with keys: tier, color, emoji, recommendation.
    """
    if score >= 75:
        return {
            "tier": "Strong Match",
            "color": "#22c55e",
            "emoji": "✅",
            "recommendation": (
                "Excellent alignment. Proceed to interview. "
                "Candidate demonstrates strong keyword and skill overlap."
            ),
        }
    elif score >= 50:
        return {
            "tier": "Moderate Match",
            "color": "#f59e0b",
            "emoji": "⚠️",
            "recommendation": (
                "Reasonable fit with gaps. Consider a screening call to assess "
                "missing competencies before a full interview."
            ),
        }
    else:
        return {
            "tier": "Weak Match",
            "color": "#ef4444",
            "emoji": "❌",
            "recommendation": (
                "Significant skill gap detected. Resume does not sufficiently "
                "align with the job requirements as described."
            ),
        }
