"""
Caseware Customer Voice — NLP Sentiment & Pain Point Analysis
================================================================
Analyzes customer reviews from G2, Capterra, Software Advice, and Reddit
to surface pain points and actionable product insights for the
Professional Services team.

Built by Hammad Mirza | Portfolio Project for Caseware PS Consultant Role
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS
from collections import Counter
import matplotlib.pyplot as plt
import re
import os
import io

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Caseware Customer Voice — NLP Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    .main { background-color: #f8f9fc; }
    .header-banner {
        background: linear-gradient(135deg, #0f2b46 0%, #1a5276 50%, #0f2b46 100%);
        padding: 2.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white; text-align: center;
    }
    .header-banner h1 { font-family: 'Source Sans Pro', sans-serif; font-weight: 700; font-size: 2rem; margin-bottom: 0.3rem; color: white; }
    .header-banner p { font-family: 'Source Sans Pro', sans-serif; font-weight: 300; font-size: 1.05rem; opacity: 0.9; margin: 0; }
    .section-header {
        font-family: 'Source Sans Pro', sans-serif; font-weight: 700; font-size: 1.25rem;
        color: #0f2b46; border-bottom: 3px solid #1a5276; padding-bottom: 0.4rem;
        margin-top: 1.5rem; margin-bottom: 1rem;
    }
    .insight-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid #1a5276; margin-bottom: 0.8rem;
    }
    .pain-flag {
        background: #fff3f0; border-left: 4px solid #e74c3c; padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0; margin-bottom: 0.5rem; font-size: 0.9rem;
    }
    .strength-flag {
        background: #f0faf4; border-left: 4px solid #27ae60; padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0; margin-bottom: 0.5rem; font-size: 0.9rem;
    }
    .footer-note {
        text-align: center; color: #888; font-size: 0.8rem; margin-top: 2rem;
        padding-top: 1rem; border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>📊 Caseware Customer Voice</h1>
    <p>NLP-Driven Sentiment & Pain Point Analysis — G2 | Capterra | Software Advice | Reddit</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
> **Purpose:** This tool analyzes 221 customer reviews across two data layers — **verified reviews** 
> (G2, Capterra, Software Advice) with structured pros/cons and star ratings, and **community 
> sentiment** (Reddit) filtered to Caseware-relevant discussions — to surface pain points, product 
> strengths, and actionable insights for the Professional Services team.
""")


# ─────────────────────────────────────────────────────────────
# STOP WORDS (shared across topic modeling and word clouds)
# ─────────────────────────────────────────────────────────────

# Comprehensive stop word list: English + review platform artifacts + generic terms
CUSTOM_STOPS = set(ENGLISH_STOP_WORDS) | {
    # Generic review/product language
    'caseware', 'software', 'product', 'tool', 'program', 'application', 'app',
    'use', 'using', 'used', 'user', 'users', 'review', 'reviews', 'reviewer',
    'recommend', 'recommended', 'overall', 'experience', 'rating', 'ratings',
    'good', 'great', 'best', 'better', 'worst', 'nice', 'excellent', 'amazing',
    'like', 'really', 'well', 'just', 'much', 'lot', 'lot', 'able', 'also',
    'would', 'could', 'should', 'might', 'may', 'shall', 'will', 'can',
    'make', 'made', 'makes', 'get', 'got', 'gets', 'know', 'known',
    'think', 'want', 'need', 'needs', 'way', 'ways', 'thing', 'things',
    'time', 'times', 'year', 'years', 'month', 'months', 'week', 'day',
    'work', 'working', 'works', 'worked', 'come', 'comes', 'came',
    'take', 'takes', 'took', 'find', 'found', 'give', 'gives', 'gave',
    'tell', 'look', 'looks', 'try', 'tried', 'trying',
    'new', 'old', 'first', 'last', 'right', 'sure', 'even', 'still',
    'though', 'however', 'although', 'said', 'say', 'says', 'see', 'seen',
    'going', 'go', 'goes', 'went', 'done', 'doing', 'did', 'does',
    'put', 'set', 'let', 'run', 'end', 'far', 'bit', 'big', 'bad',
    'back', 'long', 'keep', 'kept', 'start', 'started', 'point',
    'actually', 'pretty', 'quite', 'enough',
    # Scraping artifacts
    'collected', 'hosted', 'com', 'thumbnail', 'sidebar', 'promoted',
    'post', 'image', 'avatar', 'show', 'details', 'read', 'click',
    'sign', 'join', 'log', 'continue', 'skip', 'content', 'navigation',
    'reply', 'upvote', 'downvote', 'share', 'comments', 'comment',
    'ago', 'deleted', 'removed', 'reddit', 'subreddit',
    'source', 'verified', 'linkedin', 'profile',
    # Platform boilerplate
    'validated', 'incentivized', 'invite', 'emp', 'guest', 'upvotes',
    'conversation', 'community', 'ask', 'asked',
    'answers', 'real', 'honest', 'pros', 'cons', 'website', 'learn',
    'information', 'comparisons', 'practical', 'recommendations',
    'alternatives', 'view', 'explore', 'workflows', 'discussions',
    'currently', 'available', 'visit', 'fewer',
    'less', 'more', 'reviewed', 'employees', 'management',
    'none', 'full', 'helpful', 'pricing',
    # Foreign fragments
    'el', 'em', 'en', 'fr', 'je', 'la', 'las', 'los', 'que', 'se',
    'uso', 'uma', 'da', 'es', 'muy', 'por', 'para', 'mas', 'nos',
    'con', 'del', 'les', 'des', 'une',
    # Short noise
    'gu', 'orb', 'vpa', 'vr', 'bin', 'lol', 'hey', 'ok', 'yes',
    'guy', 'guys', 'fan', 'pay', 'job', 'tr', 'ic',
    # G2 specific
    'g2', 'capterra',
}


# ─────────────────────────────────────────────────────────────
# DATA LOADING & PROCESSING
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_and_process_data():
    """Load reviews CSV and run VADER sentiment analysis."""
    possible_paths = ['reviews.csv', os.path.join(os.path.dirname(__file__), 'reviews.csv')]
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    if df is None:
        st.error("reviews.csv not found.")
        st.stop()

    df['text'] = df['text'].fillna('').astype(str)
    df['pros'] = df['pros'].fillna('').astype(str)
    df['cons'] = df['cons'].fillna('').astype(str)
    df = df[df['text'].str.len() > 20].reset_index(drop=True)

    analyzer = SentimentIntensityAnalyzer()
    sentiments = df['text'].apply(lambda x: analyzer.polarity_scores(x))
    df['vader_compound'] = sentiments.apply(lambda x: x['compound'])
    df['sentiment'] = df['vader_compound'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )
    return df


@st.cache_data
def extract_topics(texts, n_topics=6, n_words=8):
    """Extract topics using TF-IDF + NMF with thorough stop word removal."""
    valid_texts = [t for t in texts if len(t) > 30]
    if len(valid_texts) < 10:
        return [], None

    vectorizer = TfidfVectorizer(
        max_df=0.80,
        min_df=3,
        max_features=800,
        stop_words=list(CUSTOM_STOPS),
        ngram_range=(1, 2),
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()

        nmf = NMF(n_components=min(n_topics, len(valid_texts) - 1),
                  random_state=42, max_iter=500)
        W = nmf.fit_transform(tfidf_matrix)

        topics = []
        for topic in nmf.components_:
            top_indices = topic.argsort()[-n_words:][::-1]
            words = [feature_names[i] for i in top_indices]
            # Filter duplicate bigrams like "accounting accounting"
            words = [w for w in words if len(w.split()) == 1 or w.split()[0] != w.split()[-1]]
            topics.append(words)

        return topics, W
    except Exception:
        return [], None


def extract_pain_points(df):
    """Extract and categorize pain points from negative reviews and cons."""
    pain_categories = {
        'User Interface & Usability': [
            'interface', 'clunky', 'unintuitive', 'confusing', 'navigate',
            'navigation', 'cluttered', 'dated', 'modern', 'layout',
            'user friendly', 'cumbersome', 'complicated', 'complex',
            'overwhelming', 'intuitive', 'clumsy', 'awkward', 'frustrating'
        ],
        'Learning Curve & Training': [
            'learning curve', 'steep', 'training', 'difficult to learn',
            'onboarding', 'documentation', 'tutorial', 'manual',
            'figure out', 'getting started', 'hard to understand'
        ],
        'Performance & Speed': [
            'slow', 'lag', 'loading', 'freeze', 'crash', 'hang', 'speed',
            'performance', 'unresponsive', 'timeout', 'sluggish'
        ],
        'Cloud & Migration': [
            'cloud', 'migration', 'migrate', 'desktop', 'transition', 'upgrade',
            'version', 'cloudbridge', 'cloud version', 'desktop version'
        ],
        'Cost & Licensing': [
            'expensive', 'cost', 'price', 'license', 'licensing',
            'subscription', 'fee', 'money', 'overpriced'
        ],
        'Customer Support': [
            'support', 'customer service', 'help desk', 'ticket',
            'response time', 'technical support'
        ],
        'Features & Functionality': [
            'feature', 'functionality', 'missing', 'lack', 'limited',
            'customization', 'customize', 'flexibility', 'integration',
            'export', 'import', 'template', 'formatting'
        ],
        'Updates & Bugs': [
            'bug', 'error', 'glitch', 'fix', 'update', 'patch',
            'broken', 'not working', 'fails', 'problem', 'malfunction'
        ]
    }

    negative_texts = df[df['sentiment'] == 'Negative']['text'].tolist()
    cons_texts = df[df['cons'].str.len() > 5]['cons'].tolist()
    all_complaints = negative_texts + cons_texts

    results = {}
    for category, keywords in pain_categories.items():
        matches = []
        for text in all_complaints:
            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords):
                matches.append(text)
        results[category] = {
            'count': len(matches),
            'percentage': round(len(matches) / max(len(all_complaints), 1) * 100, 1),
            'samples': matches[:3]
        }

    return dict(sorted(results.items(), key=lambda x: x[1]['count'], reverse=True))


def label_topic(words):
    """Auto-generate a descriptive label based on topic word patterns."""
    joined = ' '.join([w.lower() for w in words])

    if any(w in joined for w in ['workiva', 'vendor']):
        return "Competitive landscape"
    elif any(w in joined for w in ['das', 'onpoint', 'suite']):
        return "DAS & OnPoint ecosystem"
    elif 'trial balance' in joined:
        return "Trial balance & tax workflows"
    elif 'customer service' in joined:
        return "Customer service experience"
    elif any(w in joined for w in ['cloud', 'desktop', 'version', 'migration']):
        return "Cloud vs. desktop migration"
    elif any(w in joined for w in ['data', 'import', 'extract', 'idea']):
        return "Data import & analytics (IDEA)"
    elif any(w in joined for w in ['training', 'curve', 'steep']):
        return "Learning curve & training"
    elif any(w in joined for w in ['engagement', 'firm', 'cch']):
        return "Engagement workflows"
    elif any(w in joined for w in ['excel', 'functionality', 'feature']):
        return "Features & Excel comparison"
    elif any(w in joined for w in ['audit', 'paper', 'financial']):
        return "Audit & financial reporting"
    elif any(w in joined for w in ['price', 'cost', 'license', 'subscription']):
        return "Pricing & licensing"
    elif any(w in joined for w in ['interface', 'navigate', 'clunky']):
        return "User interface & usability"
    elif any(w in joined for w in ['tax', 'returns', 'preparation']):
        return "Tax preparation"
    else:
        return None  # Skip topics that can't be meaningfully labeled


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
df = load_and_process_data()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")
    sources = st.multiselect(
        "Filter by Source",
        options=df['source'].unique().tolist(),
        default=df['source'].unique().tolist()
    )
    sentiment_filter = st.multiselect(
        "Filter by Sentiment",
        options=['Positive', 'Neutral', 'Negative'],
        default=['Positive', 'Neutral', 'Negative']
    )

    st.markdown("---")
    st.markdown("### 📐 Methodology")
    st.caption("""
    **Sentiment:** VADER — lexicon-based tool scoring each review from -1 (negative) to +1 (positive).
    
    **Pain Points:** Keyword categorization of negative reviews and cons into 8 actionable categories.
    
    **Topics:** TF-IDF + NMF — unsupervised extraction of latent themes from review text.
    
    **Reddit:** Filtered from 279 raw comments to 94 Caseware-relevant discussions using keyword matching.
    """)
    st.markdown("---")
    st.caption("Built by **Hammad Mirza**")
    st.caption("Portfolio Project - April 2026")

filtered_df = df[df['source'].isin(sources) & df['sentiment'].isin(sentiment_filter)]
if len(filtered_df) == 0:
    st.warning("No reviews match the selected filters.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Reviews", len(filtered_df))
with col2:
    st.metric("Sources", len(filtered_df['source'].unique()))
with col3:
    avg_sentiment = filtered_df['vader_compound'].mean()
    st.metric("Avg Sentiment", f"{avg_sentiment:.2f}",
              delta="Positive" if avg_sentiment > 0 else "Negative")
with col4:
    rated = filtered_df[filtered_df['rating'] > 0]
    avg_rating = rated['rating'].mean() if len(rated) > 0 else 0
    st.metric("Avg Rating", f"{avg_rating:.1f}/5" if avg_rating > 0 else "N/A")
with col5:
    neg_pct = round(len(filtered_df[filtered_df['sentiment'] == 'Negative']) / len(filtered_df) * 100, 1)
    st.metric("Negative %", f"{neg_pct}%")

# Source breakdown
with st.expander("Data source breakdown"):
    source_summary = filtered_df.groupby('source').agg(
        Reviews=('text', 'count'),
        Avg_Sentiment=('vader_compound', 'mean'),
        Positive=('sentiment', lambda x: (x == 'Positive').sum()),
        Neutral=('sentiment', lambda x: (x == 'Neutral').sum()),
        Negative=('sentiment', lambda x: (x == 'Negative').sum()),
    ).round(3)
    source_summary['Type'] = source_summary.index.map(
        lambda x: 'Community' if x == 'Reddit' else 'Verified'
    )
    source_summary = source_summary[['Type', 'Reviews', 'Avg_Sentiment', 'Positive', 'Neutral', 'Negative']]
    st.dataframe(source_summary, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# SENTIMENT BY SOURCE
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Sentiment by Source</div>', unsafe_allow_html=True)

col_s1, col_s2 = st.columns(2)

with col_s1:
    sent_counts = filtered_df['sentiment'].value_counts()
    colors_map = {'Positive': '#27ae60', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
    fig_pie = go.Figure(data=[go.Pie(
        labels=sent_counts.index, values=sent_counts.values,
        marker_colors=[colors_map.get(s, '#999') for s in sent_counts.index],
        hole=0.4, textinfo='label+percent', textfont_size=13
    )])
    fig_pie.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Source Sans Pro"),
                          showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_s2:
    source_sent = filtered_df.groupby('source')['vader_compound'].mean().sort_values()
    bar_colors = ['#e74c3c' if v < -0.05 else '#27ae60' if v > 0.05 else '#f39c12'
                  for v in source_sent.values]
    fig_bar = go.Figure(go.Bar(
        x=source_sent.values, y=source_sent.index, orientation='h',
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in source_sent.values], textposition='auto'
    ))
    fig_bar.update_layout(height=280, margin=dict(l=10, r=30, t=10, b=10),
                          xaxis=dict(title="Mean Sentiment Score", range=[-1, 1]),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Source Sans Pro", size=13))
    st.plotly_chart(fig_bar, use_container_width=True)

st.caption("Reddit (community) and Capterra show lower sentiment than G2. G2 reviews are often incentivized, while Reddit captures unfiltered practitioner opinions.")


# ─────────────────────────────────────────────────────────────
# WORD CLOUDS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">What Customers Talk About</div>', unsafe_allow_html=True)

wc_stops = set(WC_STOPWORDS) | CUSTOM_STOPS

col_wc1, col_wc2 = st.columns(2)

with col_wc1:
    st.markdown("**Positive reviews**")
    pos_text = ' '.join(filtered_df[filtered_df['sentiment'] == 'Positive']['text'].tolist())
    if len(pos_text) > 50:
        wc_pos = WordCloud(width=600, height=300, background_color='white',
                           colormap='Greens', max_words=60, stopwords=wc_stops,
                           collocations=False, min_word_length=3).generate(pos_text)
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.imshow(wc_pos, interpolation='bilinear')
        ax1.axis('off')
        plt.tight_layout(pad=0)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf1.seek(0)
        st.image(buf1, use_container_width=True)
        plt.close(fig1)

with col_wc2:
    st.markdown("**Negative reviews & cons**")
    neg_text = ' '.join(
        filtered_df[filtered_df['sentiment'] == 'Negative']['text'].tolist() +
        filtered_df[filtered_df['cons'].str.len() > 5]['cons'].tolist()
    )
    if len(neg_text) > 50:
        wc_neg = WordCloud(width=600, height=300, background_color='white',
                           colormap='Reds', max_words=60, stopwords=wc_stops,
                           collocations=False, min_word_length=3).generate(neg_text)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.imshow(wc_neg, interpolation='bilinear')
        ax2.axis('off')
        plt.tight_layout(pad=0)
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf2.seek(0)
        st.image(buf2, use_container_width=True)
        plt.close(fig2)


# ─────────────────────────────────────────────────────────────
# PAIN POINT ANALYSIS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Customer Pain Points</div>', unsafe_allow_html=True)
st.caption("Categorized from negative reviews and cons sections. A single review can match multiple categories.")

pain_points = extract_pain_points(filtered_df)

pain_df = pd.DataFrame([
    {'Category': cat, 'Mentions': data['count'], 'Percentage': data['percentage']}
    for cat, data in pain_points.items() if data['count'] > 0
]).sort_values('Mentions', ascending=True)

if len(pain_df) > 0:
    fig_pain = go.Figure(go.Bar(
        x=pain_df['Mentions'], y=pain_df['Category'], orientation='h',
        marker_color='#e74c3c',
        text=[f"{m} ({p}%)" for m, p in zip(pain_df['Mentions'], pain_df['Percentage'])],
        textposition='auto', textfont=dict(color='white', size=12)
    ))
    fig_pain.update_layout(
        height=350, margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(title="Mentions in Negative Reviews / Cons"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans Pro", size=13)
    )
    st.plotly_chart(fig_pain, use_container_width=True)

# Top pain points with quotes
top_pains = [(cat, data) for cat, data in pain_points.items() if data['count'] > 0][:4]
for category, data in top_pains:
    st.markdown(f'<div class="pain-flag"><strong>{category}</strong> — {data["count"]} mentions ({data["percentage"]}%)</div>', unsafe_allow_html=True)
    for sample in data['samples'][:1]:
        clean = sample[:200] + "..." if len(sample) > 200 else sample
        st.caption(f'    > "{clean}"')


# ─────────────────────────────────────────────────────────────
# TOPIC MODELING
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Key Themes in Reviews (NMF Topic Modeling)</div>', unsafe_allow_html=True)
st.caption("Themes discovered through unsupervised machine learning. Each topic is auto-labeled based on its dominant keywords.")

topics, W = extract_topics(filtered_df['text'].tolist(), n_topics=8)

if topics and W is not None:
    topic_counts = pd.Series(W.argmax(axis=1)).value_counts().sort_index()
    displayed = 0

    for i, topic_words in enumerate(topics):
        topic_name = label_topic(topic_words)

        # Skip unlabelable topics (noise)
        if topic_name is None:
            continue

        words_str = " · ".join(topic_words[:5])
        review_count = topic_counts.get(i, 0)

        # Get average sentiment for this topic
        topic_indices = [idx for idx, t in enumerate(W.argmax(axis=1)) if t == i]
        if topic_indices:
            topic_sent = filtered_df.iloc[topic_indices]['vader_compound'].mean()
            if topic_sent >= 0.3:
                sent_icon = "🟢"
            elif topic_sent >= 0:
                sent_icon = "🟡"
            else:
                sent_icon = "🔴"
            sent_str = f" · Sentiment: {sent_icon} {topic_sent:.2f}"
        else:
            sent_str = ""

        st.markdown(f"""<div class="insight-card">
            <strong style="color: #1a5276;">{topic_name}</strong>
            <span style="color: #888; font-size: 0.85rem;">({review_count} reviews){sent_str}</span><br>
            <span style="color: #666; font-size: 0.85rem;">Keywords: {words_str}</span>
        </div>""", unsafe_allow_html=True)
        displayed += 1

    if displayed == 0:
        st.caption("No clear topics could be extracted from the current filter selection.")


# ─────────────────────────────────────────────────────────────
# PRODUCT STRENGTHS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Product Strengths</div>', unsafe_allow_html=True)
st.caption("What customers consistently praise, extracted from positive reviews and pros sections.")

positive_texts = filtered_df[filtered_df['sentiment'] == 'Positive']['text'].tolist()
pros_texts = filtered_df[filtered_df['pros'].str.len() > 5]['pros'].tolist()
all_positive = positive_texts + pros_texts

strength_categories = {
    'Comprehensive Audit Platform': ['comprehensive', 'complete', 'all in one', 'robust', 'powerful', 'everything'],
    'Document Organization': ['organize', 'organization', 'structured', 'repository', 'document', 'filing'],
    'Financial Reporting': ['financial statement', 'reporting', 'financial report', 'gifi', 'statements'],
    'Collaboration': ['collaborate', 'collaboration', 'team', 'real time', 'together'],
    'Cloud Accessibility': ['cloud', 'anywhere', 'remote', 'access', 'browser', 'online'],
    'Automation & Efficiency': ['automate', 'automation', 'automatic', 'efficient', 'time saving', 'streamline'],
    'Compliance & Standards': ['compliance', 'standard', 'cas', 'aspe', 'ifrs', 'regulatory', 'methodology'],
    'Trial Balance & Working Papers': ['trial balance', 'working paper', 'lead sheet', 'engagement', 'workpaper']
}

strengths_found = []
for category, keywords in strength_categories.items():
    matches = sum(1 for text in all_positive if any(kw in text.lower() for kw in keywords))
    if matches > 2:
        strengths_found.append((category, matches))

strengths_found.sort(key=lambda x: x[1], reverse=True)
for category, count in strengths_found:
    st.markdown(f'<div class="strength-flag">🟢 <strong>{category}</strong> — mentioned positively in {count} reviews</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PS TEAM IMPLICATIONS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Implications for Professional Services</div>', unsafe_allow_html=True)

st.markdown("""
<div class="insight-card">
<h4 style="margin-top: 0; color: #0f2b46;">How This Data Should Inform Implementation Strategy</h4>
</div>
""", unsafe_allow_html=True)

insights = []

if pain_points.get('User Interface & Usability', {}).get('count', 0) > 3:
    insights.append("**UI/UX is the #1 concern.** Consultants should dedicate extra training time to workflow shortcuts, customization options, and interface navigation. First impressions during onboarding set the tone for adoption.")

if pain_points.get('Learning Curve & Training', {}).get('count', 0) > 3:
    insights.append("**Learning curve is a recurring barrier.** Engagements should budget adequate training time in the SOW and consider phased rollouts — power users first, then broader team with peer mentoring support.")

if pain_points.get('Cloud & Migration', {}).get('count', 0) > 2:
    insights.append("**Cloud migration generates friction.** Firms transitioning from desktop Working Papers need structured change management. Highlight collaboration and remote access benefits while acknowledging the adjustment period.")

if pain_points.get('Cost & Licensing', {}).get('count', 0) > 2:
    insights.append("**Cost concerns surface in reviews.** Consultants should ensure value realization is tracked during implementation — ROI becomes visible when firms see reduced engagement cycle time and lower rework rates.")

if pain_points.get('Customer Support', {}).get('count', 0) > 2:
    insights.append("**Support experience is inconsistent.** Thorough knowledge transfer during implementation reduces the firm's dependence on post-go-live support tickets.")

reddit_df = filtered_df[filtered_df['source'] == 'Reddit']
if len(reddit_df) > 0:
    reddit_mean = reddit_df['vader_compound'].mean()
    g2_df = filtered_df[filtered_df['source'] == 'G2']
    g2_mean = g2_df['vader_compound'].mean() if len(g2_df) > 0 else 0
    if reddit_mean < g2_mean - 0.2:
        insights.append(f"**Community sentiment ({reddit_mean:.2f}) is significantly lower than verified reviews ({g2_mean:.2f}).** Practitioners on Reddit and forums express more frustration than formal review platforms capture. New users may arrive with negative preconceptions — proactive onboarding and early wins are essential.")

if not insights:
    insights.append("No significant patterns detected with the current filter selection.")

for insight in insights:
    st.markdown(f"- {insight}")


# ─────────────────────────────────────────────────────────────
# REVIEW EXPLORER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Review Explorer</div>', unsafe_allow_html=True)

with st.expander("Browse individual reviews"):
    display_df = filtered_df[['source', 'rating', 'sentiment', 'vader_compound', 'text']].copy()
    display_df['vader_compound'] = display_df['vader_compound'].round(3)
    display_df.columns = ['Source', 'Rating', 'Sentiment', 'VADER Score', 'Review Text']
    sort_by = st.selectbox("Sort by", ['VADER Score', 'Rating', 'Source'])
    ascending = st.checkbox("Ascending", value=True)
    display_df = display_df.sort_values(sort_by, ascending=ascending)
    st.dataframe(display_df, use_container_width=True, height=400)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer-note">
    <strong>Caseware Customer Voice</strong> · NLP Sentiment & Pain Point Analysis · v1.2<br>
    Built by Hammad Mirza · Professional Services Portfolio Project · April 2026<br>
    <em>Data sourced from public reviews on G2, Capterra, Software Advice, and Reddit.
    Not affiliated with or endorsed by Caseware International Inc.</em><br>
    <em>Methodology: VADER Sentiment · TF-IDF + NMF Topic Modeling · Keyword Pain Point Categorization</em>
</div>
""", unsafe_allow_html=True)
