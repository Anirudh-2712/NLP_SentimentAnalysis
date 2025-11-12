# data_cleaner.py
import re
import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- NEW TRANSFORMER IMPORT ---
from transformers import pipeline


def download_nltk_data():
    """
    Downloads the necessary NLTK data packages.
    """
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK 'punkt_tab' tokenizer...")
        # --- FIXED ---
        nltk.download('punkt_tab')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK 'stopwords'...")
        nltk.download('stopwords')
    
    try:
        nltk.data.find('sentiment/vader_lexicon.xml')
    except LookupError:
        print("Downloading NLTK 'vader_lexicon'...")
        nltk.download('vader_lexicon')
# ---------------------------------

# --- VADER Functions (Unchanged) ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(cleaned_tokens)

def get_vader_sentiment(text):
    if not text:
        return 0.0
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

# --- LOCAL TRANSFORMER FUNCTIONS ---

@st.cache_resource
def load_sentiment_model():
    """
    Loads and caches the sentiment analysis pipeline.
    """
    print("Loading sentiment model...")
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # Force it to use CPU (device=-1) for maximum compatibility
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=-1)
    print("Sentiment model loaded.")
    return sentiment_pipeline

def get_transformer_sentiment(text, sentiment_pipeline):
    """
    Analyzes a single piece of text and returns a sentiment score.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    truncated_text = text[:512] 
    
    try:
        result = sentiment_pipeline(truncated_text)
        result_data = result[0]
        score = result_data['score']
        label = result_data['label']
        
        # --- THIS IS THE FIX ---
        # The model returns lowercase labels, so we check for lowercase.
        
        if label == 'positive':
            return score
        elif label == 'negative':
            return -score
        else: # The label is 'neutral'
            return 0.0
        # --- END OF FIX ---
            
    except Exception as e:
        print(f"!!! ERROR processing text: {e}") 
        return 0.0