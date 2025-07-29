import numpy as np
import pandas as pd
from typing import Optional, List
import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Load environment variables
load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# News API key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if not NEWS_API_KEY:
    logger.error("NEWS_API_KEY not found in environment variables")

# Ensure NLTK data
nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

# News cache
news_cache = {"timestamp": None, "articles": None}
NEWS_CACHE_DURATION = timedelta(minutes=480)

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
def fetch_latest_news(top_n: int = 10) -> Optional[List[dict]]:
    """
    Fetch latest Bitcoin news articles in English with retry logic.
    Returns up to top_n articles.
    """
    try:
        current_time = datetime.now()
        if news_cache["timestamp"] and (current_time - news_cache["timestamp"]) < NEWS_CACHE_DURATION:
            logger.debug("Using cached news articles")
            return news_cache["articles"][:top_n]

        if not NEWS_API_KEY:
            logger.error("No News API key available")
            return None

        logger.info("Fetching latest cryptocurrency news...")
        url = f"https://newsapi.org/v2/everything?q=cryptocurrency&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        articles = response.json().get('articles', [])
        
        news_cache["articles"] = articles
        logger.info(f"Fetched {len(articles)} articles")
        
        if logger.isEnabledFor(logging.DEBUG):
            for article in articles[:top_n]:
                logger.debug(f"Article: {article.get('title', 'No Title')} - {article.get('url', 'No URL')}")

        for article in articles[:top_n]:
            article['publishedAt'] = article.get('publishedAt', '').split('T')[0]  # Keep only the date
            logger.info(f"Fetched article: {article.get('title', 'No Title')} - {article.get('url', 'No URL')}")
        return articles[:top_n]
    
    except requests.RequestException as e:
        logger.error(f"Failed to fetch news: {e}")
        return None

def calculate_sentiment(articles: Optional[List[dict]]) -> float:
    """Calculate average sentiment score from articles."""
    if not articles:
        logger.warning("No articles for sentiment analysis; returning neutral")
        return 0.0

    total_sentiment = 0.0
    for article in articles:
        text = f"{article.get('title', '')}. {article.get('description', '') or ''}"
        score = sid.polarity_scores(text)['compound']
        total_sentiment += score
    
    avg_sentiment = total_sentiment / len(articles)
    logger.info(f"Average sentiment: {avg_sentiment:.3f} (from {len(articles)} articles)")
    return avg_sentiment

def calculate_moving_average(prices: List[float], window: int) -> Optional[float]:
    """Calculate simple moving average."""
    if len(prices) < window:
        logger.debug(f"Insufficient data for MA{window}: {len(prices)} < {window}")
        return None
    return float(np.mean(prices[-window:]))

def calculate_rsi(prices: List[float], window: int = 14) -> Optional[float]:
    """Calculate Relative Strength Index."""
    if len(prices) < window + 1:
        logger.debug(f"Insufficient data for RSI{window}: {len(prices)} < {window + 1}")
        return None
    delta = np.diff(prices)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def calculate_macd(prices: List[float], short_window: int = 12, long_window: int = 26,
                  signal_window: int = 9) -> Optional[tuple[float, float]]:
    """Calculate MACD and signal line."""
    if len(prices) < long_window:
        logger.debug(f"Insufficient data for MACD: {len(prices)} < {long_window}")
        return None, None
    prices_series = pd.Series(prices)
    short_ema = prices_series.ewm(span=short_window, adjust=False).mean()
    long_ema = prices_series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return float(macd.iloc[-1]), float(signal.iloc[-1])

def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Optional[tuple[float, float, float]]:
    """Calculate Bollinger Bands (upper, middle, lower)."""
    if len(prices) < period:
        logger.debug(f"Insufficient data for Bollinger Bands (period={period}): {len(prices)} < {period}")
        return None, None, None
    middle_band = float(np.mean(prices[-period:]))
    std = np.std(prices[-period:])
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    logger.debug(f"Bollinger Bands - Upper: {upper_band:.2f}, Middle: {middle_band:.2f}, Lower: {lower_band:.2f}")
    return upper_band, middle_band, lower_band

def calculate_vwap(prices: List[float], volumes: List[float]) -> Optional[float]:
    """Calculate Volume Weighted Average Price."""
    if len(prices) < 2 or len(volumes) < 2 or len(prices) != len(volumes):
        logger.debug(f"Insufficient data for VWAP: Prices={len(prices)}, Volumes={len(volumes)}")
        return None
    try:
        prices = [float(p) for p in np.array(prices).flatten()]
        volumes = [float(v) for v in np.array(volumes).flatten()]
        if len(prices) != len(volumes):
            logger.debug("Mismatched lengths after flattening")
            return None
        cum_volume = np.cumsum(volumes)
        if cum_volume[-1] == 0:
            logger.debug("Zero cumulative volume")
            return None
        vwap = np.sum(np.array(prices) * np.array(volumes)) / cum_volume[-1]
        return float(vwap)
    except Exception as e:
        logger.error(f"VWAP calculation failed: {e}")
        return None