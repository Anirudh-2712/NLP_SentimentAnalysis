import streamlit as st
import pandas as pd
from reddit_scraper import scrape_reddit

# --- MODIFIED LINE ---
# We still import clean_text for VADER, but add our new functions
from data_cleaner import (
    clean_text, 
    download_nltk_data, 
    get_vader_sentiment, 
    load_sentiment_model,          # <-- NEW
    get_transformer_sentiment      # <-- NEW
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Reddit Sentiment Analyzer",
    page_icon="🤖",
    layout="wide")

# --- NLTK Data Download ---
@st.cache_resource
def init_nltk():
    download_nltk_data()
    return True

init_nltk()

# --- NEW: LOAD THE TRANSFORMER MODEL ---
# This calls the cached function from data_cleaner.py
sentiment_model_pipeline = load_sentiment_model()

# --- Main Application ---
st.title("🤖 Reddit Data Scraper & Analyzer")
st.markdown("Enter a keyword to scrape data, clean it, and analyze sentiment.")

# --- User Input ---
with st.sidebar:
    st.header("Search Controls")
    keyword = st.text_input("Enter a keyword:", "Python")
    subreddit = st.text_input("Subreddit:", "all")
    limit = st.slider("Number of posts to fetch:", 10, 200, 50)
    
    # --- NEW: Model Selector ---
    model_choice = st.radio(
        "Choose a Sentiment Model:",
        ("VADER (Fast, Rule-Based)", "Transformer (Slow, Accurate)")
    )
        
    scrape_button = st.button("Scrape, Clean & Analyze")

# --- Data Processing and Display ---
if scrape_button:
    if not keyword:
        st.error("Please enter a keyword to start.")
    else:
        # 1. Scrape Data
        with st.spinner(f"Scraping '{subreddit}' for '{keyword}'..."):
            raw_data_df = scrape_reddit(keyword, subreddit, limit)
                
        if raw_data_df.empty:
            st.warning("No posts found.")
        else:
            st.success(f"Successfully scraped {len(raw_data_df)} posts.")
            
            # 2. Analyze Sentiment
            analysis_df = raw_data_df.copy()
            
            if model_choice == "VADER (Fast, Rule-Based)":
                st.info("Using VADER model for analysis...")
                with st.spinner("Cleaning and analyzing sentiment (VADER)..."):
                    # VADER needs cleaned text
                    analysis_df['cleaned_title'] = analysis_df['title'].apply(clean_text)
                    analysis_df['sentiment'] = analysis_df['cleaned_title'].apply(get_vader_sentiment)
                    display_text_col = 'cleaned_title'
            
            else: # Transformer
                st.info("Using Transformer model (RoBERTa)... This may take a moment.")
                with st.spinner("Analyzing sentiment (Transformer)..."):
                    # Transformer needs RAW text
                    # We use a lambda function to pass the loaded model pipeline
                    analysis_df['sentiment'] = analysis_df['title'].apply(
                        lambda text: get_transformer_sentiment(text, sentiment_model_pipeline)
                    )
                    display_text_col = 'title' # Show the raw title

            st.success("Analysis complete!")

            # 3. Display Data
            st.subheader("Sentiment Analysis")
            avg_sentiment = analysis_df['sentiment'].mean()
            col1, col2 = st.columns(2)
            col1.metric(label="Average Title Sentiment", value=f"{avg_sentiment:.3f}")
            col2.markdown("*(Score: -1.0 Negative to +1.0 Positive)*")

            st.subheader("Analyzed Data")
            st.dataframe(analysis_df[[
                'sentiment', 
                display_text_col, # Show either raw or cleaned title
                'score', 
                'num_comments', 
                'url'
            ]])
                        
            with st.expander("Show Raw Scraped Data"):
                st.dataframe(raw_data_df)
else:
    st.info("Enter your search terms in the sidebar and click 'Scrape, Clean & Analyze' to begin.")