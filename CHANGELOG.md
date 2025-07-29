# Changelog

All notable changes to the Crypto Trading Bot project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup with core trading logic in `money_bot.py`.
- Technical indicators and sentiment analysis in `indicators.py`.
- Configuration settings in `config.py`.
- Trade logging to `trades.csv`.
- Backtesting and performance reporting features.
- Triangular arbitrage checks on Kraken.
- AI model integration with Ollama for trading decisions.

### Fixed
- Addressed premature sells at small losses by planning to add unrealized profit to AI prompt (pending implementation).

## [0.1.0] - 2025-07-29

### Added
- Initial release of the bot.
- Support for trading on Bitvavo and Kraken exchanges.
- Diversified trading pairs including meme coins (BONK, DOGE, SHIB, PEPE, etc.).
- Risk management with stop-loss (3%) and take-profit (10%).
- News API integration for sentiment analysis.
- README.md for project overview and setup instructions.