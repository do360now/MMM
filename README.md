# Crypto Trading Bot

## Overview
This is an automated cryptocurrency trading bot designed to trade on centralized exchanges (Bitvavo and Kraken) using technical indicators, news sentiment analysis, and an AI model (Ollama with Gemma3:4B) to make data-driven buy and sell decisions. The bot aims to maximize EUR cashflow by trading diversified pairs, including meme coins (e.g., BONK, DOGE, SHIB, PEPE), with risk management features like stop-loss and take-profit. It also supports triangular arbitrage opportunities on Kraken and includes backtesting and performance reporting capabilities.

## Features
- **Technical Indicators**: Uses RSI, MACD, Bollinger Bands, and VWAP to inform trading decisions.
- **Sentiment Analysis**: Fetches and analyzes cryptocurrency news via News API to gauge market sentiment.
- **AI-Driven Decisions**: Integrates with Ollama (Gemma3:4B model) for adaptive trading signals based on indicators and past performance.
- **Arbitrage**: Checks for triangular arbitrage opportunities on Kraken using high-liquidity pairs.
- **Risk Management**: Implements stop-loss (3%) and take-profit (10%) thresholds, with adjustable risk per trade (1-5%).
- **Backtesting**: Simulates trading strategies over historical data to estimate ROI.
- **Performance Reporting**: Generates detailed reports on realized/unrealized PnL, win rate, and open positions.
- **Logging**: Comprehensive logging to track trades, errors, and performance metrics.

## Files
- **indicators.py**: Contains functions for calculating technical indicators (RSI, MACD, Bollinger Bands, VWAP) and fetching/analyzing news sentiment.
- **money_bot.py**: Core trading bot logic, including exchange interactions, trading decisions, backtesting, and performance reporting.
- **config.py**: Configuration file for exchange credentials, trading pairs, risk parameters, and Ollama settings.
- **trades.csv**: Logs executed trades with timestamp, type (buy/sell), pair, amount, and price.
- **authenticate.py**: (Not provided but referenced) Handles exchange authentication using API keys.
- **logger_config.py**: (Not provided but referenced) Configures logging for the bot.

## Requirements
- Python 3.8+
- Dependencies: `ccxt`, `numpy`, `pandas`, `requests`, `python-dotenv`, `tenacity`, `nltk`
- API keys for Bitvavo, Kraken, and News API
- Ollama server running locally (or accessible) with Gemma3:4B model
- Environment file (`.env`) with the following variables:
  ```
  KRAKEN_API_KEY=your_kraken_api_key
  KRAKEN_API_SECRET=your_kraken_api_secret
  BITVAVO_API_KEY=your_bitvavo_api_key
  BITVAVO_API_SECRET=your_bitvavo_api_secret
  NEWS_API_KEY=your_news_api_key
  CHECK_INTERVAL=5
  MIN_TRADE_AMOUNT=5.0
  MAX_TRADE_AMOUNT=0.0
  LOG_FILE=./triangular_arbitrage_bot.log
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies:
   ```bash
   pip install ccxt numpy pandas requests python-dotenv tenacity nltk
   ```
3. Set up NLTK data:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```
4. Configure the `.env` file with your API keys and settings.
5. Ensure Ollama is running locally:
   ```bash
   ollama serve
   ollama pull gemma3:4b
   ```
6. Run the bot:
   ```bash
   python money_bot.py
   ```

## Configuration
Edit `config.py` to customize:
- **Trading Pairs**: Modify `PAIRS` for different assets (e.g., `['BONK/EUR', 'DOGE/EUR', ...]`).
- **Arbitrage Triangles**: Adjust `TRIANGLES` for Kraken arbitrage (e.g., `('USDT', 'BTC', 'ETH')`).
- **Risk Parameters**:
  - `RISK_PER_TRADE`: 1% of portfolio per trade.
  - `AGGRESSIVE_RISK`: 5% for strong AI signals.
  - `STOP_LOSS`: 3% loss trigger.
  - `TAKE_PROFIT`: 10% profit trigger.
- **Indicator Settings**: Tune `RSI_PERIOD`, `RSI_BUY`, `RSI_SELL`, `BTC_DIP`, etc.
- **Ollama**: Set `OLLAMA_URL` and `OLLAMA_MODEL` for AI model integration.

## Usage
- **Live Trading**: Run `python money_bot.py` to start trading with real funds.
- **Simulation Mode**: Run `python money_bot.py --simulate` to test without executing real trades.
- **Backtesting**: The bot automatically runs a backtest on startup to estimate ROI over historical data (1000 periods, 5-minute candles).
- **Performance Monitoring**: Check `trades.csv` for trade history and review logs (`triangular_arbitrage_bot.log`) for detailed output.

## Trading Strategy
The bot uses a multi-signal approach:
1. **Indicators**: RSI (<30 for buy, >70 for sell), MACD crossovers, Bollinger Bands (price below/above bands), VWAP, and volume spikes.
2. **Sentiment**: Positive (>0.2) or negative (<-0.2) news sentiment influences buy/sell decisions.
3. **AI Model**: Ollama evaluates indicators, sentiment, and past performance to output "buy", "sell", or "hold".
4. **Risk Management**: Limits trades to 1-5% of portfolio, with stop-loss (3%) and take-profit (10%) triggers.
5. **Arbitrage**: Checks for triangular arbitrage on Kraken every ~5 minutes.
6. **Adaptation**: Adjusts RSI buy threshold dynamically based on win rate (<50% win rate increases `RSI_BUY` up to 50).

## Known Issues
- **Premature Sells at Losses**: The AI model may sell at small losses due to missing per-position unrealized profit data in the prompt. Consider adding this data to enforce a >5% profit rule for sells.
- **Volatile Assets**: Trading meme coins (BONK, SHIB, PEPE) leads to noisy indicators, triggering quick sells. Use longer timeframes (e.g., 15m) for stability.
- **Performance Report**: Simplified FIFO matching in reports may misrepresent open positions. Enhance with precise entry tracking.

## Future Improvements
- Add per-position unrealized profit to AI prompt for profit-focused sells.
- Implement trailing stops to capture larger gains.
- Use longer OHLCV timeframes (e.g., 15m) to reduce noise.
- Enhance backtesting with real AI model queries instead of random decisions.
- Add more robust performance metrics (e.g., Sharpe ratio, max drawdown).

## Disclaimer
This bot trades real funds and carries significant financial risk. Cryptocurrency markets are highly volatile, and losses can exceed initial capital. Use at your own risk, and test thoroughly in simulation mode before live trading. Ensure API keys are secure and comply with exchange terms of service.

## License
This project is licensed under the MIT License.