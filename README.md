# 📄 Smart Resume Screening System

> An interactive NLP web app that instantly scores how well a resume matches a job description.

🔗 **Built with** Streamlit | Scikit-learn | NLTK | PyPDF2 | Plotly

---

## 📊 Features

- 🔮 **TF-IDF + Cosine Similarity** scoring (0–100%)
- 📈 **Interactive gauge & bar charts** (Plotly)
- 🟢🟡🔴 **Tier classification** — Strong / Moderate / Weak match
- 🗝 **Keyword gap analysis** — matched vs. missing keywords from the JD
- 🗂 **Resume section detection** — Skills, Experience, Education, Projects
- ⚡ **Real-time results** with a dark-mode, modern UI

---

## 🧠 How It Works

```
Resume PDF  ──► extract_text_from_pdf()  ──► clean_text()
                                                    │
Job Description  ───────────────────────────────────┤
                                                    ▼
                                        TfidfVectorizer (1-gram + 2-gram)
                                                    │
                                        cosine_similarity()
                                                    │
                                        Score (0–100%) + Keyword Analysis
```

1. PDF text is extracted with **PyPDF2**
2. Both texts are cleaned and tokenized
3. **TF-IDF vectors** are built with unigrams and bigrams
4. **Cosine similarity** between the two vectors gives the match score
5. Keyword overlap identifies skill gaps instantly

---

## 📁 Project Structure

```
smart_resume_screening/
├── app.py              # Streamlit web application
├── resume_parser.py    # PDF extraction + text cleaning + section detection
├── model.py            # TF-IDF similarity + keyword gap analysis + classifier
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Installation

**Clone or download the project folder, then:**

```bash
# 1. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (auto-handled on first run, or manually):
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🌍 Deployment (Streamlit Community Cloud)

1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point to `app.py` in your repo
4. Click **Deploy** — no extra configuration needed

---

## 📌 Score Interpretation

| Score | Tier | Recommendation |
|-------|------|----------------|
| ≥ 75% | ✅ Strong Match | Proceed to interview |
| 50–74% | ⚠️ Moderate Match | Screening call recommended |
| < 50% | ❌ Weak Match | Significant skill gap |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| NLP | Scikit-learn (TF-IDF) |
| PDF Parsing | PyPDF2 |
| Text Processing | NLTK, Regex |
| Visualization | Plotly |
| Language | Python 3.9+ |

---

## 👨‍💻 Author

**Your Name** — Aspiring ML/NLP Engineer 🤖

⭐ If you found this useful, give it a star on GitHub!
