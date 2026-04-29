# 📊 Caseware Customer Voice — NLP Sentiment & Pain Point Analysis

**An NLP-driven analysis of Caseware customer reviews across G2, Capterra, Software Advice, and Reddit to surface sentiment trends, pain points, and actionable product insights for the Professional Services team.**

> Built as a portfolio project by Hammad Mirza — April 2026

---

## Overview

Customer reviews contain unstructured signal that rarely makes it into implementation planning. This tool bridges that gap by applying Natural Language Processing to 400+ customer reviews and community discussions about Caseware products, transforming raw feedback into structured, actionable intelligence.

For a Professional Services consultant, understanding what customers praise and where they struggle is the difference between a reactive implementation and a proactive one.

## What It Does

The tool provides five layers of analysis:

- **Sentiment Analysis (VADER):** Scores every review on a -1 to +1 compound scale, validated against star ratings where available
- **Pain Point Categorization:** Classifies negative reviews and cons into actionable categories (UI/UX, Learning Curve, Performance, Cloud Migration, Cost, Support, Features, Bugs)
- **Topic Modeling (TF-IDF + NMF):** Extracts latent topic clusters to surface themes the keyword categories might miss
- **Product Strengths:** Identifies what customers consistently praise, from collaboration features to compliance support
- **PS Team Implications:** Translates the data into specific recommendations for how consultants should adjust their implementation approach

## Data Sources

| Source | Type | Count |
|--------|------|-------|
| G2 | Verified reviews with star ratings | ~60 reviews |
| Capterra | Verified reviews with Pros/Cons | ~33 reviews |
| Software Advice | Verified reviews with Pros/Cons | ~33 reviews |
| Reddit (r/Accounting, r/Audit) | Community discussions | ~280 comments |

## Methodology

**Sentiment:** VADER (Valence Aware Dictionary and sEntiment Reasoner) — a lexicon and rule-based tool specifically tuned for social media and review text. Chosen over transformer-based models (RoBERTa) because the dataset size (~400 reviews) doesn't warrant the complexity, and VADER's performance on review text is well-validated.

**Topic Modeling:** TF-IDF vectorization followed by NMF (Non-negative Matrix Factorization). NMF produces more interpretable topics than LDA for short-text corpora, and the TF-IDF representation handles the mixed vocabulary of technical accounting terms and casual review language.

**Pain Point Extraction:** Keyword-based categorization into eight product/service categories. Each category uses a curated keyword list developed from an initial manual review of the corpus. This approach is transparent and easily extendable.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ensure `reviews.csv` is in the same directory as `app.py`.

## Tech Stack

- **Python 3.10+**
- **Streamlit** — Interactive dashboard
- **VADER** — Sentiment analysis
- **scikit-learn** — TF-IDF + NMF topic modeling
- **Plotly** — Visualizations

## Disclaimer

This is an independent portfolio project. Data was collected from publicly available reviews. It is not affiliated with, endorsed by, or connected to Caseware International Inc.

---

**Author:** Hammad Mirza
**Date:** April 2026
