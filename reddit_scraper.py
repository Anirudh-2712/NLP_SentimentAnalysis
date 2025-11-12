import praw
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_reddit_instance():
    """
    Initializes and returns a PRAW Reddit instance using credentials
    from environment variables.
    """
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )
        # Test the connection
        reddit.user.me() 
        return reddit
    except Exception as e:
        print(f"Error connecting to Reddit: {e}")
        return None

def scrape_reddit(keyword, subreddit_name="all", limit=50):
    """
    Scrapes a specified subreddit for submissions matching a keyword.
    
    Args:
        keyword (str): The search term.
        subreddit_name (str): The subreddit to search in (e.g., "all", "python").
        limit (int): The max number of posts to return.

    Returns:
        pandas.DataFrame: A DataFrame containing the scraped data.
    """
    reddit = get_reddit_instance()
    if not reddit:
        return pd.DataFrame() # Return empty DataFrame if connection failed

    try:
        # Search the subreddit
        subreddit = reddit.subreddit(subreddit_name)
        search_results = subreddit.search(keyword, limit=limit)
        
        # Store data
        data_list = []
        for submission in search_results:
            data_list.append({
                "post_id": submission.id,
                "title": submission.title,
                "body": submission.selftext, # The text content of the post
                "score": submission.score,
                "url": submission.url,
                "num_comments": submission.num_comments,
                "created_utc": submission.created_utc,
            })
            
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        return df

    except Exception as e:
        print(f"An error occurred during scraping: {e}")
        return pd.DataFrame()
