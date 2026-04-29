"""
Caseware Customer Voice — NLP Sentiment & Pain Point Analysis
================================================================
Analyzes customer reviews from G2, Capterra, Software Advice, and Reddit
to surface sentiment trends, pain points, and actionable product insights
for the Professional Services team.

Built by Hammad Mirza | Portfolio Project for Caseware PS Consultant Role
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os
import io

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
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
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .header-banner h1 {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.3rem;
        color: white;
    }
    .header-banner p {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 300;
        font-size: 1.05rem;
        opacity: 0.9;
        margin: 0;
    }

    .section-header {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        color: #0f2b46;
        border-bottom: 3px solid #1a5276;
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .insight-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #1a5276;
        margin-bottom: 0.8rem;
    }

    .pain-flag {
        background: #fff3f0;
        border-left: 4px solid #e74c3c;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }

    .strength-flag {
        background: #f0faf4;
        border-left: 4px solid #27ae60;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }

    .footer-note {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
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
> **Purpose:** This tool uses Natural Language Processing to analyze customer feedback about 
> Caseware products across two data layers: **verified reviews** (G2, Capterra, Software Advice) 
> with structured pros/cons and star ratings, and **community sentiment** (Reddit r/Accounting, 
> r/Audit) filtered to Caseware-relevant discussions only. It surfaces sentiment trends, 
> recurring pain points, and product strengths to support data-informed implementation strategies 
> for the Professional Services team.
""")


# ─────────────────────────────────────────────────────────────
# DATA LOADING & PROCESSING
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_and_process_data():
    """Load reviews CSV and run VADER sentiment analysis."""
    # Try multiple paths
    possible_paths = [
        'reviews.csv',
        os.path.join(os.path.dirname(__file__), 'reviews.csv'),
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    
    if df is None:
        st.error("reviews.csv not found. Please ensure it's in the same directory as the app.")
        st.stop()
    
    # Clean text
    df['text'] = df['text'].fillna('').astype(str)
    df['pros'] = df['pros'].fillna('').astype(str)
    df['cons'] = df['cons'].fillna('').astype(str)
    
    # Remove very short entries
    df = df[df['text'].str.len() > 20].reset_index(drop=True)
    
    # VADER Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    
    sentiments = df['text'].apply(lambda x: analyzer.polarity_scores(x))
    df['vader_compound'] = sentiments.apply(lambda x: x['compound'])
    df['vader_pos'] = sentiments.apply(lambda x: x['pos'])
    df['vader_neg'] = sentiments.apply(lambda x: x['neg'])
    df['vader_neu'] = sentiments.apply(lambda x: x['neu'])
    
    # Classify sentiment
    df['sentiment'] = df['vader_compound'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )
    
    # Separate analysis for pros and cons where available
    df['cons_sentiment'] = df['cons'].apply(
        lambda x: analyzer.polarity_scores(x)['compound'] if len(x) > 5 else 0
    )
    
    return df


@st.cache_data
def extract_topics(texts, n_topics=8, n_words=8):
    """Extract topics using TF-IDF + NMF."""
    # Custom stop words relevant to review context
    custom_stops = [
        'caseware', 'software', 'use', 'using', 'used', 'product', 'tool',
        'like', 'really', 'also', 'would', 'could', 'one', 'get', 'got',
        'much', 'make', 'made', 'well', 'good', 'great', 'just', 'know',
        'think', 'want', 'need', 'way', 'thing', 'time', 'year', 'years',
        'lot', 'able', 'work', 'working', 'even', 'still', 'though',
        'overall', 'review', 'recommend', 'rating', 'experience', 'best',
        'worst', 'better', 'however', 'although', 'said', 'say', 'see',
        'going', 'come', 'take', 'find', 'give', 'tell', 'look', 'try',
        'new', 'old', 'first', 'last', 'sure', 'right', 'don', 'didn',
        'isn', 'wasn', 'doesn', 'aren', 've', 'll', 'haven', 'hasn',
        'shouldn', 'won', 'wouldn', 'couldn', 'let',
        # Scraping artifacts
        'collected', 'hosted', 'com', 'thumbnail', 'sidebar', 'promoted',
        'post', 'image', 'avatar', 'show', 'details', 'read', 'click',
        'sign', 'join', 'log', 'continue', 'skip', 'content', 'navigation',
        'reply', 'upvote', 'downvote', 'share', 'comments', 'comment',
        'ago', 'deleted', 'removed', 'reddit', 'subreddit',
        'reviewer', 'source', 'verified', 'user', 'linkedin',
        # More platform artifacts
        'validated', 'incentivized', 'invite', 'emp', 'guest', 'upvotes',
        'conversation', 'pricing', 'community', 'ask', 'profile',
        'helpful', 'month', 'months', 'week', 'weeks', 'day', 'days',
        # Foreign language fragments (Spanish/Portuguese reviews on G2)
        'el', 'em', 'en', 'fr', 'je', 'la', 'las', 'los', 'que', 'se',
        'uso', 'uma', 'da', 'es', 'muy', 'por', 'para', 'mas', 'nos',
        'con', 'del', 'su', 'al', 'lo', 'les', 'des', 'une',
        # G2 platform boilerplate
        'answers', 'real', 'honest', 'pros', 'cons', 'website', 'learn',
        'information', 'comparisons', 'practical', 'recommendations',
        'alternatives', 'g2', 'capterra',
        # G2 boilerplate phrases
        'view', 'explore', 'workflows', 'discussions', 'currently',
        'available', 'visit', 'users', 'fewer',
        # Remaining scraping noise
        'gu', 'orb', 'vpa', 'vr', 'bin', 'lol', 'hey', 'ok', 'yes',
        'guy', 'guys', 'fan', 'bit', 'big', 'bad', 'end', 'far',
        'mid', 'non', 'pre', 'set', 'add', 'cut', 'did', 'does',
        'pay', 'job', 'tr', 'ic'
    ]
    
    # Filter out very short texts
    valid_texts = [t for t in texts if len(t) > 30]
    
    if len(valid_texts) < 10:
        return [], None, None
    
    vectorizer = TfidfVectorizer(
        max_df=0.85,
        min_df=2,
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'  # Only alphabetic tokens, 3+ chars
    )
    
    # Add custom stops
    if vectorizer.stop_words:
        vectorizer.stop_words = list(vectorizer.stop_words) + custom_stops
    else:
        vectorizer.stop_words = custom_stops
    
    try:
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        nmf = NMF(n_components=min(n_topics, len(valid_texts) - 1), 
                  random_state=42, max_iter=500)
        W = nmf.fit_transform(tfidf_matrix)
        H = nmf.components_
        
        topics = []
        for topic_idx, topic in enumerate(H):
            top_word_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            topics.append(top_words)
        
        return topics, W, feature_names
    except Exception as e:
        st.warning(f"Topic extraction encountered an issue: {str(e)}")
        return [], None, None


def extract_pain_points(df):
    """Extract and categorize pain points from negative reviews and cons."""
    pain_categories = {
        'User Interface & Usability': [
            'interface', 'ui', 'clunky', 'unintuitive', 'confusing', 'navigate', 
            'navigation', 'cluttered', 'dated', 'ugly', 'modern', 'layout',
            'user friendly', 'cumbersome', 'complicated', 'complex', 'overwhelming',
            'intuitive', 'clumsy', 'awkward', 'frustrating'
        ],
        'Learning Curve': [
            'learning curve', 'steep', 'training', 'learn', 'difficult to learn',
            'onboarding', 'documentation', 'tutorial', 'help', 'manual',
            'figure out', 'getting started', 'hard to understand'
        ],
        'Performance & Speed': [
            'slow', 'lag', 'loading', 'freeze', 'crash', 'hang', 'speed',
            'performance', 'buffer', 'unresponsive', 'timeout', 'waiting',
            'takes forever', 'sluggish'
        ],
        'Cloud & Migration': [
            'cloud', 'migration', 'migrate', 'desktop', 'transition', 'upgrade',
            'version', 'cloudbridge', 'online', 'web version', 'browser',
            'cloud version', 'moving to cloud'
        ],
        'Cost & Licensing': [
            'expensive', 'cost', 'price', 'pricing', 'license', 'licensing',
            'subscription', 'fee', 'value', 'money', 'budget', 'worth',
            'overpriced', 'affordable'
        ],
        'Customer Support': [
            'support', 'customer service', 'help desk', 'ticket', 'response time',
            'waiting for support', 'technical support', 'service', 'rep',
            'call', 'chat', 'email support'
        ],
        'Features & Functionality': [
            'feature', 'functionality', 'missing', 'lack', 'limited', 'basic',
            'customization', 'customize', 'flexibility', 'integration',
            'export', 'import', 'report', 'template', 'formatting'
        ],
        'Updates & Bugs': [
            'bug', 'error', 'glitch', 'issue', 'fix', 'update', 'patch',
            'broken', 'crash', 'not working', 'fails', 'problem',
            'malfunction', 'defect'
        ]
    }
    
    results = {}
    negative_texts = df[df['sentiment'] == 'Negative']['text'].tolist()
    cons_texts = df[df['cons'].str.len() > 5]['cons'].tolist()
    
    all_complaint_texts = negative_texts + cons_texts
    
    for category, keywords in pain_categories.items():
        matches = []
        for text in all_complaint_texts:
            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords):
                matches.append(text)
        results[category] = {
            'count': len(matches),
            'percentage': round(len(matches) / max(len(all_complaint_texts), 1) * 100, 1),
            'sample_quotes': matches[:3]
        }
    
    return dict(sorted(results.items(), key=lambda x: x[1]['count'], reverse=True))


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

df = load_and_process_data()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Analysis Controls")
    
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
    
    n_topics = st.slider("Number of Topics", min_value=4, max_value=12, value=7)
    
    st.markdown("---")
    
    st.markdown("### 📐 Methodology")
    st.caption("""
    **Sentiment:** VADER (Valence Aware Dictionary and sEntiment Reasoner) — 
    a lexicon and rule-based tool optimized for social media and review text.
    
    **Topics:** TF-IDF vectorization + NMF (Non-negative Matrix Factorization) — 
    extracts latent topic clusters from review text.
    
    **Pain Points:** Keyword-based categorization of negative reviews and cons 
    into actionable product and service categories.
    
    **Reddit Filtering:** Raw Reddit threads were filtered to retain only 
    comments containing Caseware-specific keywords (caseware, working papers, 
    DAS, trial balance, etc.) with a minimum length threshold, removing 
    platform noise and off-topic discussions.
    """)
    
    st.markdown("---")
    st.caption("Built by **Hammad Mirza**")
    st.caption("Portfolio Project - April 2026")


# Apply filters
filtered_df = df[df['source'].isin(sources) & df['sentiment'].isin(sentiment_filter)]

if len(filtered_df) == 0:
    st.warning("No reviews match the selected filters.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# OVERVIEW METRICS
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

# Data source breakdown
with st.expander("📋 Data source breakdown"):
    source_summary = filtered_df.groupby('source').agg(
        Reviews=('text', 'count'),
        Avg_Sentiment=('vader_compound', 'mean'),
        Positive=('sentiment', lambda x: (x == 'Positive').sum()),
        Neutral=('sentiment', lambda x: (x == 'Neutral').sum()),
        Negative=('sentiment', lambda x: (x == 'Negative').sum()),
        Avg_Rating=('rating', lambda x: x[x > 0].mean() if (x > 0).any() else 0)
    ).round(3)
    source_summary['Data Type'] = source_summary.index.map(
        lambda x: 'Community Sentiment' if x == 'Reddit' else 'Verified Reviews'
    )
    source_summary = source_summary[['Data Type', 'Reviews', 'Avg_Sentiment', 'Positive', 'Neutral', 'Negative', 'Avg_Rating']]
    source_summary.columns = ['Data Type', 'Count', 'Avg Sentiment', 'Positive', 'Neutral', 'Negative', 'Avg Rating']
    st.dataframe(source_summary, use_container_width=True)
    st.caption("**Verified Reviews** have structured pros/cons and star ratings. **Community Sentiment** (Reddit) was filtered to Caseware-relevant comments only, with no star ratings available.")


# ─────────────────────────────────────────────────────────────
# SENTIMENT DISTRIBUTION
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Sentiment Analysis</div>', unsafe_allow_html=True)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("##### Sentiment Distribution")
    sent_counts = filtered_df['sentiment'].value_counts()
    colors = {'Positive': '#27ae60', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=sent_counts.index,
        values=sent_counts.values,
        marker_colors=[colors.get(s, '#999') for s in sent_counts.index],
        hole=0.4,
        textinfo='label+percent',
        textfont_size=13
    )])
    fig_pie.update_layout(
        height=300, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans Pro")
    )
    st.plotly_chart(fig_pie, use_container_width=True)


with col_chart2:
    st.markdown("##### Sentiment by Source")
    source_sent = filtered_df.groupby('source')['vader_compound'].mean().sort_values()
    
    bar_colors = ['#e74c3c' if v < -0.05 else '#27ae60' if v > 0.05 else '#f39c12' 
                  for v in source_sent.values]
    
    fig_bar = go.Figure(go.Bar(
        x=source_sent.values,
        y=source_sent.index,
        orientation='h',
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in source_sent.values],
        textposition='auto'
    ))
    fig_bar.update_layout(
        height=300, margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(title="Mean VADER Compound Score", range=[-1, 1]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans Pro", size=13)
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# Sentiment score distribution histogram
st.markdown("##### VADER Compound Score Distribution")
fig_hist = go.Figure()
for sent, color in [('Negative', '#e74c3c'), ('Neutral', '#f39c12'), ('Positive', '#27ae60')]:
    subset = filtered_df[filtered_df['sentiment'] == sent]
    fig_hist.add_trace(go.Histogram(
        x=subset['vader_compound'],
        name=sent,
        marker_color=color,
        opacity=0.7,
        nbinsx=30
    ))
fig_hist.update_layout(
    barmode='overlay',
    height=250,
    margin=dict(l=30, r=30, t=10, b=30),
    xaxis=dict(title="Compound Score (-1 = most negative, +1 = most positive)"),
    yaxis=dict(title="Count"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Source Sans Pro", size=12),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
st.plotly_chart(fig_hist, use_container_width=True)


# Rating vs VADER validation
rated_df = filtered_df[filtered_df['rating'] > 0]
if len(rated_df) > 10:
    st.markdown("##### Star Rating vs. VADER Sentiment Score")
    st.caption("Validates VADER accuracy — sentiment scores should correlate with star ratings.")
    fig_scatter = px.scatter(
        rated_df, x='rating', y='vader_compound',
        color='source',
        opacity=0.6,
        labels={'rating': 'Star Rating', 'vader_compound': 'VADER Compound Score'},
        color_discrete_sequence=['#1a5276', '#27ae60', '#f39c12', '#e74c3c']
    )
    fig_scatter.update_layout(
        height=280, margin=dict(l=30, r=30, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans Pro", size=12)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    corr = rated_df[['rating', 'vader_compound']].corr().iloc[0, 1]
    st.caption(f"Pearson correlation between star rating and VADER score: **{corr:.3f}**")


# ── Word Clouds ──
st.markdown("##### Word Clouds — Positive vs. Negative Reviews")
st.caption("Most frequent distinctive words in positive and negative reviews. Larger words appear more often.")

# Custom stop words for word clouds
from wordcloud import STOPWORDS as WC_STOPWORDS
wc_stopwords = set(WC_STOPWORDS) | {
    'caseware', 'software', 'use', 'using', 'used', 'product', 'tool', 'also',
    'would', 'could', 'one', 'get', 'got', 'really', 'like', 'much', 'make',
    'made', 'well', 'good', 'great', 'just', 'know', 'think', 'want', 'need',
    'way', 'thing', 'time', 'year', 'years', 'lot', 'able', 'work', 'working',
    'even', 'still', 'though', 'overall', 'review', 'recommend', 'experience',
    'best', 'worst', 'better', 'however', 'said', 'say', 'see', 'going', 'come',
    'take', 'find', 'give', 'tell', 'look', 'try', 'new', 'old', 'first', 'last',
    'sure', 'right', 'many', 'every', 'will', 'can', 'may', 'might', 'shall',
    'must', 'need', 'back', 'go', 'do', 'done', 'put', 'two', 'three', 'per',
    # Scraping artifacts
    'collected', 'hosted', 'validated', 'incentivized', 'answers', 'real',
    'honest', 'pros', 'cons', 'information', 'comparisons', 'practical',
    'currently', 'available', 'visit', 'explore', 'discussions', 'users',
    'reviewer', 'source', 'verified', 'user', 'profile', 'fewer',
    'com', 'emp', 'invite', 'guest', 'upvotes', 'conversation',
    'view', 'workflows', 'accounting', 'small', 'business',
    'review collected', 'review verified',
    'less', 'read', 'show', 'more', 'reviewed', 'employees',
    'management', 'none', 'full'
}

col_wc1, col_wc2 = st.columns(2)

with col_wc1:
    st.markdown("**Positive reviews**")
    positive_text = ' '.join(filtered_df[filtered_df['sentiment'] == 'Positive']['text'].tolist())
    if len(positive_text) > 50:
        wc_pos = WordCloud(
            width=600, height=350, background_color='white',
            colormap='Greens', max_words=80, stopwords=wc_stopwords,
            collocations=False, min_word_length=3
        ).generate(positive_text)
        fig_wc1, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.imshow(wc_pos, interpolation='bilinear')
        ax1.axis('off')
        plt.tight_layout(pad=0)
        buf1 = io.BytesIO()
        fig_wc1.savefig(buf1, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf1.seek(0)
        st.image(buf1, use_container_width=True)
        plt.close(fig_wc1)

with col_wc2:
    st.markdown("**Negative reviews & cons**")
    negative_text = ' '.join(
        filtered_df[filtered_df['sentiment'] == 'Negative']['text'].tolist() +
        filtered_df[filtered_df['cons'].str.len() > 5]['cons'].tolist()
    )
    if len(negative_text) > 50:
        wc_neg = WordCloud(
            width=600, height=350, background_color='white',
            colormap='Reds', max_words=80, stopwords=wc_stopwords,
            collocations=False, min_word_length=3
        ).generate(negative_text)
        fig_wc2, ax2 = plt.subplots(figsize=(6, 3.5))
        ax2.imshow(wc_neg, interpolation='bilinear')
        ax2.axis('off')
        plt.tight_layout(pad=0)
        buf2 = io.BytesIO()
        fig_wc2.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf2.seek(0)
        st.image(buf2, use_container_width=True)
        plt.close(fig_wc2)


# ── Divergent Reviews ──
if len(rated_df) > 10:
    st.markdown("##### Divergent Reviews — Where Ratings and Sentiment Disagree")
    st.caption("Reviews where star ratings tell a different story than the text. These are often the most insightful for understanding hidden frustrations or unexpected praise.")
    
    col_div1, col_div2 = st.columns(2)
    with col_div1:
        neg_threshold = st.slider(
            "VADER threshold for 'negative text'", 
            min_value=-0.5, max_value=0.0, value=-0.05, step=0.05,
            help="Reviews with VADER score below this are considered negative text"
        )
    with col_div2:
        pos_threshold = st.slider(
            "VADER threshold for 'positive text'", 
            min_value=0.0, max_value=0.8, value=0.3, step=0.05,
            help="Reviews with VADER score above this are considered positive text"
        )
    
    # High stars but negative text
    high_star_neg = rated_df[(rated_df['rating'] >= 4) & (rated_df['vader_compound'] < neg_threshold)].copy()
    high_star_neg = high_star_neg.sort_values('vader_compound', ascending=True)
    
    # Low stars but positive text
    low_star_pos = rated_df[(rated_df['rating'] <= 3) & (rated_df['vader_compound'] > pos_threshold)].copy()
    low_star_pos = low_star_pos.sort_values('vader_compound', ascending=False)
    
    col_dv1, col_dv2 = st.columns(2)
    
    with col_dv1:
        if len(high_star_neg) > 0:
            st.markdown(f'<div class="pain-flag"><strong>High rating, negative text</strong> — {len(high_star_neg)} review{"s" if len(high_star_neg) != 1 else ""}. Users who like the product but have real frustrations.</div>', unsafe_allow_html=True)
            for _, row in high_star_neg.iterrows():
                with st.expander(f"[{row['source']}] {row['rating']}/5 stars · VADER: {row['vader_compound']:.2f}"):
                    st.write(row['text'][:500])
                    if row['pros']:
                        st.caption(f"**Pros:** {row['pros'][:200]}")
                    if row['cons']:
                        st.caption(f"**Cons:** {row['cons'][:200]}")
        else:
            st.caption("No high-rating / negative-text divergences at this threshold.")
    
    with col_dv2:
        if len(low_star_pos) > 0:
            st.markdown(f'<div class="strength-flag"><strong>Low rating, positive text</strong> — {len(low_star_pos)} review{"s" if len(low_star_pos) != 1 else ""}. Users who see value but have a blocking issue.</div>', unsafe_allow_html=True)
            for _, row in low_star_pos.iterrows():
                with st.expander(f"[{row['source']}] {row['rating']}/5 stars · VADER: {row['vader_compound']:.2f}"):
                    st.write(row['text'][:500])
                    if row['pros']:
                        st.caption(f"**Pros:** {row['pros'][:200]}")
                    if row['cons']:
                        st.caption(f"**Cons:** {row['cons'][:200]}")
        else:
            st.caption("No low-rating / positive-text divergences at this threshold.")


# ─────────────────────────────────────────────────────────────
# PAIN POINT ANALYSIS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Pain Point Analysis</div>', unsafe_allow_html=True)
st.caption("Categorized from negative reviews and 'Cons' sections across all platforms.")

pain_points = extract_pain_points(filtered_df)

# Pain point bar chart
pain_df = pd.DataFrame([
    {'Category': cat, 'Mentions': data['count'], 'Percentage': data['percentage']}
    for cat, data in pain_points.items()
    if data['count'] > 0
]).sort_values('Mentions', ascending=True)

if len(pain_df) > 0:
    fig_pain = go.Figure(go.Bar(
        x=pain_df['Mentions'],
        y=pain_df['Category'],
        orientation='h',
        marker_color='#e74c3c',
        text=[f"{m} ({p}%)" for m, p in zip(pain_df['Mentions'], pain_df['Percentage'])],
        textposition='auto',
        textfont=dict(color='white', size=12)
    ))
    fig_pain.update_layout(
        height=350, margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(title="Number of Mentions in Negative Reviews / Cons"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans Pro", size=13)
    )
    st.plotly_chart(fig_pain, use_container_width=True)

# Top pain points with sample quotes
st.markdown("##### Top Pain Points — Representative Customer Quotes")

top_pains = [(cat, data) for cat, data in pain_points.items() if data['count'] > 0][:5]
for category, data in top_pains:
    st.markdown(f'<div class="pain-flag"><strong>{category}</strong> — {data["count"]} mentions ({data["percentage"]}%)</div>', unsafe_allow_html=True)
    for quote in data['sample_quotes'][:2]:
        clean_quote = quote[:200] + "..." if len(quote) > 200 else quote
        st.caption(f'    > "{clean_quote}"')


# ─────────────────────────────────────────────────────────────
# TOPIC MODELING
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Topic Modeling (NMF)</div>', unsafe_allow_html=True)
st.caption("Latent topics extracted from all review text using TF-IDF + Non-negative Matrix Factorization. Topics are auto-labeled based on dominant word patterns.")

topics, W, feature_names = extract_topics(filtered_df['text'].tolist(), n_topics=n_topics)


def label_topic(words):
    """Auto-generate a descriptive label based on topic word patterns."""
    words_lower = [w.lower() for w in words]
    joined = ' '.join(words_lower)
    
    # Check specific patterns first, then broader ones
    if any(w in joined for w in ['workiva', 'vendor']):
        return "Competitive landscape & vendor comparison"
    elif any(w in joined for w in ['das', 'onpoint', 'suite']):
        return "DAS & OnPoint product ecosystem"
    elif any(w in joined for w in ['trial balance']):
        return "Trial balance & tax workflows"
    elif 'customer service' in joined:
        return "Excel comparison & customer service"
    elif any(w in joined for w in ['cloud', 'desktop', 'version', 'migration', 'transition']):
        return "Cloud vs. desktop migration"
    elif any(w in joined for w in ['data', 'import', 'extract', 'idea']):
        return "Data import & extraction (IDEA)"
    elif any(w in joined for w in ['training', 'learn', 'curve', 'steep']):
        return "Learning curve & training"
    elif any(w in joined for w in ['engagement', 'firm', 'cch']):
        return "Engagement & firm workflows"
    elif any(w in joined for w in ['excel', 'functionality', 'feature']):
        return "Features & Excel comparison"
    elif any(w in joined for w in ['audit', 'paper', 'financial']):
        return "Audit & financial reporting"
    elif any(w in joined for w in ['price', 'pricing', 'cost', 'license', 'subscription']):
        return "Pricing & licensing"
    elif any(w in joined for w in ['interface', 'navigate', 'clunky', 'ui']):
        return "User interface & usability"
    elif any(w in joined for w in ['tax', 'returns', 'preparation']):
        return "Tax preparation workflows"
    else:
        return "General discussion"


if topics:
    # Calculate how many reviews belong to each topic
    if W is not None:
        topic_counts = pd.Series(W.argmax(axis=1)).value_counts().sort_index()
    
    for i, topic_words in enumerate(topics):
        # Filter out duplicate-word bigrams like "accounting accounting"
        clean_words = [w for w in topic_words if len(w.split()) == 1 or w.split()[0] != w.split()[-1]]
        
        # Auto-generate label
        topic_name = label_topic(clean_words)
        words_str = " · ".join(clean_words[:6])
        
        # Get review count for this topic
        review_count = topic_counts.get(i, 0) if W is not None else ""
        count_str = f" ({review_count} reviews)" if review_count else ""
        
        # Get average sentiment for reviews in this topic
        if W is not None:
            topic_review_indices = [idx for idx, t in enumerate(W.argmax(axis=1)) if t == i]
            if topic_review_indices:
                topic_sentiment = filtered_df.iloc[topic_review_indices]['vader_compound'].mean()
                if topic_sentiment >= 0.3:
                    sent_indicator = "🟢"
                elif topic_sentiment >= 0:
                    sent_indicator = "🟡"
                else:
                    sent_indicator = "🔴"
                sent_str = f" · Avg sentiment: {sent_indicator} {topic_sentiment:.2f}"
            else:
                sent_str = ""
        else:
            sent_str = ""
        
        st.markdown(f"""<div class="insight-card">
            <strong style="color: #1a5276;">{topic_name}</strong>{count_str}{sent_str}<br>
            <span style="color: #666; font-size: 0.85rem;">Keywords: {words_str}</span>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PRODUCT STRENGTHS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Product Strengths</div>', unsafe_allow_html=True)
st.caption("Extracted from positive reviews and 'Pros' sections.")

positive_texts = filtered_df[filtered_df['sentiment'] == 'Positive']['text'].tolist()
pros_texts = filtered_df[filtered_df['pros'].str.len() > 5]['pros'].tolist()
all_positive = positive_texts + pros_texts

strength_categories = {
    'Comprehensive Audit Platform': ['comprehensive', 'complete', 'all in one', 'full suite', 'robust', 'powerful', 'everything'],
    'Document Organization': ['organize', 'organization', 'structured', 'repository', 'document', 'filing', 'manage files'],
    'Financial Reporting': ['financial statement', 'reporting', 'financial report', 'gifi', 'statements', 'balance sheet'],
    'Collaboration': ['collaborate', 'collaboration', 'team', 'share', 'real time', 'together', 'multi user'],
    'Cloud Accessibility': ['cloud', 'anywhere', 'remote', 'access', 'browser', 'online', 'web based'],
    'Automation': ['automate', 'automation', 'automatic', 'efficient', 'time saving', 'saves time', 'streamline'],
    'Compliance & Standards': ['compliance', 'standard', 'cas', 'aspe', 'ifrs', 'regulatory', 'methodology'],
    'Trial Balance & Working Papers': ['trial balance', 'working paper', 'lead sheet', 'engagement', 'workpaper']
}

for category, keywords in strength_categories.items():
    matches = sum(1 for text in all_positive if any(kw in text.lower() for kw in keywords))
    if matches > 2:
        st.markdown(f'<div class="strength-flag">🟢 <strong>{category}</strong> — mentioned positively in {matches} reviews</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PS TEAM INSIGHTS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Implications for Professional Services</div>', unsafe_allow_html=True)

st.markdown("""
<div class="insight-card">
<h4 style="margin-top: 0; color: #0f2b46;">How This Data Informs Implementation Strategy</h4>
<p style="font-size: 0.9rem; line-height: 1.7;">
The sentiment and pain point analysis reveals patterns that directly affect how PS consultants should approach client engagements:
</p>
</div>
""", unsafe_allow_html=True)

# Dynamic insights based on actual data
top_pain = list(pain_points.keys())[0] if pain_points else "N/A"
top_pain_pct = pain_points[top_pain]['percentage'] if top_pain != "N/A" else 0

insights = []

if 'User Interface & Usability' in [k for k, v in pain_points.items() if v['count'] > 3]:
    insights.append("**UI/UX concerns are frequent.** Consultants should spend extra time during training demonstrating workflow shortcuts and customization options. First impressions of the interface matter — guided walkthroughs reduce negative perception.")

if 'Learning Curve' in [k for k, v in pain_points.items() if v['count'] > 3]:
    insights.append("**Learning curve is a recurring theme.** Implementation engagements should budget adequate training time and consider a phased rollout with power users first. Quick reference guides and cheat sheets will accelerate adoption.")

if 'Performance & Speed' in [k for k, v in pain_points.items() if v['count'] > 2]:
    insights.append("**Performance issues appear in reviews.** During discovery, assess the firm's network infrastructure and browser environment. Set expectations about cloud performance and recommend optimal browser/connectivity settings.")

if 'Cloud & Migration' in [k for k, v in pain_points.items() if v['count'] > 2]:
    insights.append("**Cloud migration generates mixed sentiment.** Firms transitioning from desktop Working Papers need a structured change management plan. Highlight cloud advantages (collaboration, accessibility) while acknowledging the adjustment period.")

if 'Cost & Licensing' in [k for k, v in pain_points.items() if v['count'] > 2]:
    insights.append("**Cost concerns surface in reviews.** PS consultants should ensure clients understand the ROI and total cost of ownership. Value realization tracking during the engagement helps justify the investment.")

if 'Customer Support' in [k for k, v in pain_points.items() if v['count'] > 2]:
    insights.append("**Support experience varies.** Consultants can mitigate this by ensuring thorough knowledge transfer during implementation, reducing the firm's dependence on post-go-live support tickets.")

reddit_df = filtered_df[filtered_df['source'] == 'Reddit']
if len(reddit_df) > 0:
    reddit_neg_pct = round(len(reddit_df[reddit_df['sentiment'] == 'Negative']) / len(reddit_df) * 100, 1)
    if reddit_neg_pct > 40:
        insights.append(f"**Reddit sentiment skews negative ({reddit_neg_pct}% negative).** Community forums amplify frustration. PS consultants should be aware that new users may arrive with negative preconceptions from online discussions. Proactive onboarding and early wins counter this.")

for insight in insights:
    st.markdown(f"- {insight}")


# ─────────────────────────────────────────────────────────────
# RAW DATA EXPLORER
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
    <strong>Caseware Customer Voice</strong> · NLP Sentiment & Pain Point Analysis · v1.1<br>
    Built by Hammad Mirza · Professional Services Portfolio Project · April 2026<br>
    <em>Data sourced from public reviews on G2, Capterra, Software Advice, and Reddit.
    This tool is not affiliated with or endorsed by Caseware International Inc.</em><br>
    <em>Methodology: VADER Sentiment Analysis · TF-IDF + NMF Topic Modeling · Keyword-Based Pain Point Categorization</em>
</div>
""", unsafe_allow_html=True)
