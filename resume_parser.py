"""
resume_parser.py
----------------
Extracts and cleans text from uploaded PDF resumes.
"""

import PyPDF2
import re
import nltk

# Download required NLTK data silently
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


def extract_text_from_pdf(file) -> str:
    """
    Extract raw text from a PDF file object.

    Args:
        file: A file-like object (e.g. from Streamlit's file_uploader).

    Returns:
        Extracted text as a single string.
    """
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")
    return text


def clean_text(text: str) -> str:
    """
    Normalize and clean raw text for NLP processing.

    Steps:
        - Lowercase
        - Remove URLs
        - Remove special characters / punctuation
        - Collapse whitespace

    Args:
        text: Raw string.

    Returns:
        Cleaned string.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # Remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)            # Keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()             # Collapse whitespace
    return text


def extract_sections(text: str) -> dict:
    """
    Heuristically split a resume into common sections.

    Detected sections: skills, experience, education, projects, certifications.

    Args:
        text: Raw resume text.

    Returns:
        Dict mapping section name → content string.
    """
    section_headers = {
        "skills": r"\b(skills|technical skills|core competencies|expertise)\b",
        "experience": r"\b(experience|work experience|employment|professional experience)\b",
        "education": r"\b(education|academic background|qualifications)\b",
        "projects": r"\b(projects|personal projects|key projects)\b",
        "certifications": r"\b(certifications|certificates|licenses)\b",
    }

    lines = text.splitlines()
    sections: dict[str, list[str]] = {k: [] for k in section_headers}
    current_section = None

    for line in lines:
        line_lower = line.lower().strip()
        matched = False
        for section, pattern in section_headers.items():
            if re.search(pattern, line_lower):
                current_section = section
                matched = True
                break
        if not matched and current_section:
            sections[current_section].append(line)

    return {k: " ".join(v).strip() for k, v in sections.items()}
