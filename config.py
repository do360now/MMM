# filename: config.py
import os
from dotenv import load_dotenv
from typing import List, Tuple

# Load environment variables from the .env file
load_dotenv()

# Get API credentials from environment variables
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET")
BITVAVO_API_KEY = os.getenv("BITVAVO_API_KEY")
BITVAVO_API_SECRET = os.getenv("BITVAVO_API_SECRET")

# Ensure critical environment variables are set
required_vars = ["KRAKEN_API_KEY", "KRAKEN_API_SECRET", "BITVAVO_API_KEY", "BITVAVO_API_SECRET"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

# Triangles for arbitrage on Kraken (base: EUR, int1: BTC, int2: various high-liquidity assets)
# Chosen based on liquidity and availability: ETH/BTC, SOL/BTC, XRP/BTC, ADA/BTC
TRIANGLES: List[Tuple[str, str, str]] = [
    ('USDT', 'BTC', 'ETH'),
    ('USDT', 'BTC', 'SOL'),
    ('USDT', 'BTC', 'XRP'),
    ('USDT', 'BTC', 'ADA'),
    ('USDT', 'BTC', 'DOGE'),
]

# Approximate fees (taker fees per trade) - adjust based on your account tier
# Kraken: ~0.26% taker fee
FEE_KRAKEN: float = 0.0026

# Minimum profit threshold (e.g., 0.5% after fees for 3 trades)
MIN_PROFIT_THRESHOLD: float = 0.005

# Cooldown period in seconds between checks
CHECK_INTERVAL: int = int(os.getenv("CHECK_INTERVAL", "15"))  # Faster for triangular: 5 seconds

# Minimum trade amount (in base currency, e.g., EUR) - lowered to 5.0 based on Kraken min orders
MIN_TRADE_AMOUNT: float = float(os.getenv("MIN_TRADE_AMOUNT", "5.0"))

# Optional max trade amount (in base currency) to cap risk; set to 0 for no cap
MAX_TRADE_AMOUNT: float = float(os.getenv("MAX_TRADE_AMOUNT", "0.0"))

# Log file (used in logger_config, but defined here for consistency)
LOG_FILE: str = os.getenv("LOG_FILE", "./triangular_arbitrage_bot.log")

# Helper function
def safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value else default
    except (ValueError, TypeError):
        return default
    
# ... existing ...
PAIRS = ['BONK/EUR', 'DOGE/EUR', 'SHIB/EUR', 'PEPE/EUR', 'SOL/EUR', 'XRP/EUR', 'FLOKI/EUR', 'USDC/EUR']  # Diversified with more meme coins
VOL_THRESHOLD = 0.05  # % for dip/pump
RSI_PERIOD = 14
RSI_BUY = 30
RSI_SELL = 70
BTC_DIP = 0.02  # 2% for BTC buy
MIN_EUR_FOR_BTC = 300
FEE = 0.0026

# New: Risk management params
RISK_PER_TRADE = 0.01  # 1% of portfolio per trade base
AGGRESSIVE_RISK = 0.05  # Up to 5% on strong model signals
STOP_LOSS = 0.03  # 3% loss trigger
TAKE_PROFIT = 0.05  # 5% profit trigger

# Ollama config
OLLAMA_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'gemma3:4b'