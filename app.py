"""
app.py
------
Smart Resume Screening System — Streamlit Web App
Built with: Streamlit | Scikit-learn | NLTK | PyPDF2
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

from resume_parser import extract_text_from_pdf, extract_sections
from model import calculate_similarity, get_keyword_analysis, classify_candidate

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    /* ── Root palette ── */
    :root {
        --bg:         #0d0f14;
        --surface:    #13161e;
        --surface2:   #1b1f2b;
        --border:     #252a38;
        --accent:     #6c63ff;
        --accent2:    #00e5cc;
        --text:       #e8eaf0;
        --muted:      #6b7280;
        --green:      #22c55e;
        --amber:      #f59e0b;
        --red:        #ef4444;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero header ── */
    .hero {
        background: linear-gradient(135deg, #1b1f2b 0%, #0d0f14 60%, #12101f 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(108,99,255,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: var(--text);
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero h1 span { color: var(--accent); }
    .hero p {
        color: var(--muted);
        font-size: 1rem;
        margin: 0;
        font-weight: 300;
    }

    /* ── Cards ── */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.8rem;
    }

    /* ── Score display ── */
    .score-ring-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1rem 0;
    }
    .score-number {
        font-family: 'Syne', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin-top: -0.3rem;
    }
    .score-label {
        font-size: 0.85rem;
        color: var(--muted);
        margin-top: 0.3rem;
        letter-spacing: 1px;
    }

    /* ── Tier badge ── */
    .tier-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1.2rem;
        border-radius: 100px;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }

    /* ── Keyword pills ── */
    .pill-container { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
    .pill {
        padding: 0.3rem 0.8rem;
        border-radius: 100px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    .pill-match  { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3);  }
    .pill-miss   { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3);  }

    /* ── Metric chips ── */
    .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .metric-chip {
        flex: 1;
        min-width: 120px;
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-chip .value {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-chip .label {
        font-size: 0.72rem;
        color: var(--muted);
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 0.2rem;
    }

    /* ── Section text preview ── */
    .section-text {
        font-size: 0.82rem;
        color: var(--muted);
        line-height: 1.6;
        max-height: 100px;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Recommendation box ── */
    .reco-box {
        border-left: 3px solid var(--accent);
        padding: 0.8rem 1rem;
        background: rgba(108,99,255,0.07);
        border-radius: 0 10px 10px 0;
        font-size: 0.88rem;
        line-height: 1.6;
        color: var(--text);
        margin-top: 0.5rem;
    }

    /* ── Streamlit widget overrides ── */
    .stTextArea textarea, .stFileUploader {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), #8b83ff) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        padding: 0.6rem 2rem !important;
        letter-spacing: 0.5px !important;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }
    .stSpinner > div { border-top-color: var(--accent) !important; }

    /* Divider */
    hr { border-color: var(--border) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <h1>📄 Smart <span>Resume Screener</span></h1>
      <p>AI-powered resume–job description matching using TF-IDF NLP &amp; cosine similarity</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# INPUT COLUMNS
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card-title">📁 Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a PDF resume here",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.success(f"✔ Loaded: **{uploaded_file.name}**")

with col_right:
    st.markdown('<div class="card-title">📋 Job Description</div>', unsafe_allow_html=True)
    job_desc = st.text_area(
        "Paste the job description",
        height=180,
        placeholder="Paste the full job description here…",
        label_visibility="collapsed",
    )

st.markdown("")
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    analyze_btn = st.button("⚡ Analyze Now")

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
if analyze_btn:
    if not uploaded_file:
        st.warning("Please upload a PDF resume.")
    elif not job_desc.strip():
        st.warning("Please paste a job description.")
    else:
        with st.spinner("Running NLP analysis…"):
            time.sleep(0.4)  # brief UX pause

            # ── Extract & score ──
            resume_text = extract_text_from_pdf(uploaded_file)
            score = calculate_similarity(resume_text, job_desc)
            classification = classify_candidate(score)
            kw = get_keyword_analysis(resume_text, job_desc)
            sections = extract_sections(resume_text)

        st.markdown("---")
        st.markdown(
            '<div style="font-family:\'Syne\',sans-serif; font-size:0.7rem; '
            'letter-spacing:2px; text-transform:uppercase; color:#6c63ff; '
            'margin-bottom:1rem;">📊 Analysis Results</div>',
            unsafe_allow_html=True,
        )

        # ─────────── Row 1: Score gauge + Metrics ───────────
        r1_left, r1_right = st.columns([1, 2], gap="large")

        with r1_left:
            # Gauge chart
            gauge_color = classification["color"]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "%", "font": {"size": 38, "family": "Syne", "color": gauge_color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1,
                             "tickcolor": "#252a38", "tickfont": {"color": "#6b7280", "size": 10}},
                    "bar": {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": "#1b1f2b",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50],  "color": "rgba(239,68,68,0.08)"},
                        {"range": [50, 75], "color": "rgba(245,158,11,0.08)"},
                        {"range": [75, 100],"color": "rgba(34,197,94,0.08)"},
                    ],
                    "threshold": {
                        "line": {"color": gauge_color, "width": 3},
                        "thickness": 0.8,
                        "value": score,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=10, l=20, r=20),
                height=220,
                font_color="#e8eaf0",
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            tier_color = classification["color"]
            st.markdown(
                f'<div style="text-align:center;">'
                f'<span class="tier-badge" style="background:rgba(0,0,0,0.3);'
                f'border:1.5px solid {tier_color}; color:{tier_color};">'
                f'{classification["emoji"]} {classification["tier"]}</span></div>',
                unsafe_allow_html=True,
            )

        with r1_right:
            # Metric chips
            st.markdown(
                f"""
                <div class="metric-row">
                  <div class="metric-chip">
                    <div class="value" style="color:#6c63ff;">{score}%</div>
                    <div class="label">Match Score</div>
                  </div>
                  <div class="metric-chip">
                    <div class="value" style="color:#22c55e;">{len(kw['matched'])}</div>
                    <div class="label">Keywords Hit</div>
                  </div>
                  <div class="metric-chip">
                    <div class="value" style="color:#ef4444;">{len(kw['missing'])}</div>
                    <div class="label">Missing KWs</div>
                  </div>
                  <div class="metric-chip">
                    <div class="value" style="color:#00e5cc;">{kw['coverage']}%</div>
                    <div class="label">KW Coverage</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Recommendation
            st.markdown('<div class="card-title">🎯 Recruiter Recommendation</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="reco-box">{classification["recommendation"]}</div>',
                unsafe_allow_html=True,
            )

        # ─────────── Row 2: Keyword Analysis ───────────
        st.markdown("")
        kw_left, kw_right = st.columns(2, gap="large")

        with kw_left:
            st.markdown('<div class="card-title">✅ Matched Keywords</div>', unsafe_allow_html=True)
            if kw["matched"]:
                pills_html = "".join(
                    f'<span class="pill pill-match">{w}</span>'
                    for w in kw["matched"][:30]
                )
                st.markdown(
                    f'<div class="pill-container">{pills_html}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<span style="color:#6b7280;font-size:0.85rem;">No overlapping keywords found.</span>', unsafe_allow_html=True)

        with kw_right:
            st.markdown('<div class="card-title">❌ Missing Keywords</div>', unsafe_allow_html=True)
            if kw["missing"]:
                pills_html = "".join(
                    f'<span class="pill pill-miss">{w}</span>'
                    for w in kw["missing"][:30]
                )
                st.markdown(
                    f'<div class="pill-container">{pills_html}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<span style="color:#4ade80;font-size:0.85rem;">All JD keywords are present in the resume! 🎉</span>', unsafe_allow_html=True)

        # ─────────── Row 3: Bar chart ───────────
        st.markdown("")
        st.markdown('<div class="card-title">📈 Score Breakdown</div>', unsafe_allow_html=True)

        bar_data = pd.DataFrame({
            "Category": ["Overall Match", "Keyword Coverage"],
            "Score": [score, kw["coverage"]],
            "Color": [classification["color"], "#00e5cc"],
        })

        fig_bar = px.bar(
            bar_data,
            x="Category",
            y="Score",
            color="Category",
            color_discrete_map={
                "Overall Match": classification["color"],
                "Keyword Coverage": "#00e5cc",
            },
            text="Score",
            range_y=[0, 100],
        )
        fig_bar.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker_line_width=0,
            width=0.4,
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eaf0",
            font_family="DM Sans",
            showlegend=False,
            margin=dict(t=30, b=20, l=20, r=20),
            height=260,
            xaxis=dict(showgrid=False, tickfont=dict(size=13, family="Syne")),
            yaxis=dict(
                showgrid=True,
                gridcolor="#252a38",
                ticksuffix="%",
                tickfont=dict(size=11),
            ),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # ─────────── Row 4: Resume Section Preview ───────────
        detected = {k: v for k, v in sections.items() if v}
        if detected:
            st.markdown("")
            st.markdown('<div class="card-title">🗂 Detected Resume Sections</div>', unsafe_allow_html=True)
            sec_cols = st.columns(len(detected))
            for col, (section, content) in zip(sec_cols, detected.items()):
                with col:
                    st.markdown(
                        f"""
                        <div class="card">
                          <div class="card-title">{section.upper()}</div>
                          <div class="section-text">{content[:200]}…</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ─────────── Footer tip ───────────
        st.markdown("")
        st.markdown(
            '<div style="text-align:center; color:#6b7280; font-size:0.78rem; padding:1rem 0;">'
            '💡 Tip: Add missing keywords to your resume to improve your match score. '
            'Focus on skills and technologies explicitly listed in the job description.'
            '</div>',
            unsafe_allow_html=True,
        )
