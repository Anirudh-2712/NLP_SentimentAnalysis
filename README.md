# Reddit Sentiment Analyzer (VADER vs. RoBERTa)

This project is a Python web application built with Streamlit that scrapes live data from Reddit and performs sentiment analysis. It allows for a direct comparison between a fast, rule-based model (VADER) and a state-of-the-art transformer model (RoBERTa).

## ✨ Features

* **Reddit Data Scraping:** Fetches post titles from any subreddit using a keyword.
* **Dual-Model Analysis:** Lets the user choose between two different NLP models:
    * **VADER:** A fast, lexicon-based model (good for general sentiment).
    * **Transformer (RoBERTa):** A powerful, context-aware deep learning model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) for high-accuracy analysis.
* **Interactive UI:** Built with Streamlit for a simple, responsive user interface.

## 📸 Screenshots


<img width="940" height="502" alt="image" src="https://github.com/user-attachments/assets/f45e8f5f-3af5-449b-a887-98cd3d7cf272" />
<img width="940" height="502" alt="image" src="https://github.com/user-attachments/assets/404530f8-bb34-4bec-a435-1105654afa43" />


## 🚀 How to Run

### 1. Prerequisites

* Python 3.8+
* A Reddit Account (for API credentials)
* [Optional] A Hugging Face Account (if using the API version)

### 2. Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_PROJECT_NAME.git](https://github.com/YOUR_USERNAME/YOUR_PROJECT_NAME.git)
    cd YOUR_PROJECT_NAME
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create your API key file:**
    * Create a file in the main folder named `.env`
    * Copy the contents of `.env.example` (see below) into it and add your keys.

### 3. Running the App

Once set up, just run the following command in your terminal:

```bash
streamlit run app.py
