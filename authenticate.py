# filename: authenticate.py
import ccxt
import os
from dotenv import load_dotenv
from logger_config import logger  # Import the standardized logger
from config import KRAKEN_API_KEY, KRAKEN_API_SECRET, BITVAVO_API_KEY, BITVAVO_API_SECRET

# Load environment variables from the .env file (redundant if already loaded in config, but safe)
load_dotenv()

# Check if all API credentials are available
logger.info("Checking if all API credentials are available...")
missing_creds = [key for key, val in {"KRAKEN_API_KEY": KRAKEN_API_KEY, "KRAKEN_API_SECRET": KRAKEN_API_SECRET, 
                                      "BITVAVO_API_KEY": BITVAVO_API_KEY, "BITVAVO_API_SECRET": BITVAVO_API_SECRET}.items() if not val]
if missing_creds:
    raise ValueError(f"Missing API credentials: {', '.join(missing_creds)}")

def authenticate_exchanges():
    """
    Authenticate with Kraken and Bitvavo using CCXT and return the exchange objects.
    """
    logger.info("Authenticating with Kraken and Bitvavo via CCXT...")
    
    kraken = ccxt.kraken({
        'apiKey': KRAKEN_API_KEY,
        'secret': KRAKEN_API_SECRET,
    })
    
    bitvavo = ccxt.bitvavo({
        'apiKey': BITVAVO_API_KEY,
        'secret': BITVAVO_API_SECRET,
    })
    
    # Verify authentication by fetching balance (optional, but good for validation)
    try:
        kraken_balance = kraken.fetch_balance()
        bitvavo_balance = bitvavo.fetch_balance()
        logger.debug(f"Kraken balance: {kraken_balance}")
        logger.debug(f"Bitvavo balance: {bitvavo_balance}")
        logger.info("Successfully authenticated with both exchanges.")
    except Exception as e:
        logger.error(f"Authentication verification failed: {e}")
        raise
    
    return kraken, bitvavo