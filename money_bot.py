# filename: money_bot.py
import ccxt
import time
import numpy as np
import pandas as pd
import os
import requests  # For Ollama
from logger_config import logger
from config import TRIANGLES, FEE, VOL_THRESHOLD, RSI_PERIOD, RSI_BUY, RSI_SELL, BTC_DIP, MIN_EUR_FOR_BTC, PAIRS, RISK_PER_TRADE, AGGRESSIVE_RISK, STOP_LOSS, TAKE_PROFIT, OLLAMA_URL, OLLAMA_MODEL
from authenticate import authenticate_exchanges
from indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, fetch_latest_news, calculate_sentiment, calculate_vwap

# Authenticate exchanges
kraken, bitvavo = authenticate_exchanges()

import random

class TradingBot:
    def __init__(self, primary_exchange, secondary_exchange, btc_pair, usdc_pair, min_trade_amount=10, max_loss=0.05):
        self.primary = primary_exchange
        self.secondary = secondary_exchange
        self.pairs = [self._normalize_symbol(pair, self.primary) for pair in PAIRS]  # Diversified pairs
        self.btc_pair = self._normalize_symbol(btc_pair, self.primary)
        self.usdc_pair = self._normalize_symbol(usdc_pair, self.primary)
        self.min_trade_amount = min_trade_amount
        self.max_loss = max_loss
        self.rsi_buy = RSI_BUY
        self.rsi_sell = RSI_SELL
        self.primary.load_markets()
        self.secondary.load_markets()
        self.primary.options['operatorId'] = random.randint(1000000000, 9999999999)  # Generate random 10-digit operatorId
        self.ollama_url = OLLAMA_URL
        self.ollama_model = OLLAMA_MODEL
        self.entry_prices = {pair: 0.0 for pair in self.pairs}  # Track entry for stop/take
        logger.info(f"Initialized bot with primary {self.primary.id} (pairs: {self.pairs}), secondary {self.secondary.id}")

    def _normalize_symbol(self, symbol, exch):
        if exch.id == 'kraken':
            return symbol.replace('-', '/').replace('BTC', 'XBT')
        return symbol

    def get_ohlcv(self, exch, pair, timeframe='1m', limit=100):
        logger.debug(f"Fetching OHLCV for {pair} on {exch.id}")
        try:
            ohlcv = exch.fetch_ohlcv(pair, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df[df['close'] > 0]
            time.sleep(1)  # Rate limit buffer
            logger.debug(f"Fetched {len(df)} candles for {pair}; last close: {df['close'].iloc[-1] if not df.empty else 'N/A'}")
            return df
        except Exception as e:
            logger.error(f"OHLCV fetch failed: {e}")
            return pd.DataFrame()

    def query_model(self, prompt):
        try:
            payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()['response'].upper()
            logger.info(f"Model response: {result}")
            if 'BUY' in result:
                return 'BUY', 'STRONG' in result  # Check for confidence
            elif 'SELL' in result:
                return 'SELL', 'STRONG' in result
            else:
                return 'HOLD', False
        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
            return 'HOLD', False

    def check_arbitrage(self, triangles, min_profit=0.005):
        # Called more frequently now
        for base, int1, int2 in triangles:
            pair1 = self._normalize_symbol(f"{int1}/{base}", self.secondary)
            pair2 = self._normalize_symbol(f"{int2}/{int1}", self.secondary)
            pair3 = self._normalize_symbol(f"{int2}/{base}", self.secondary)
            if all(p in self.secondary.markets for p in [pair1, pair2, pair3]):
                try:
                    ticker1 = self.secondary.fetch_ticker(pair1)
                    ticker2 = self.secondary.fetch_ticker(pair2)
                    ticker3 = self.secondary.fetch_ticker(pair3)
                    forward = (1 / ticker1['ask']) * (1 / ticker2['ask']) * ticker3['bid']
                    reverse = ticker1['bid'] * ticker2['bid'] * (1 / ticker3['ask'])
                    profit_forward = (forward - 1) - 3 * FEE
                    profit_reverse = (reverse - 1) - 3 * FEE
                    if max(profit_forward, profit_reverse) > min_profit:
                        logger.info(f"Arbitrage opp on {base}-{int1}-{int2}: Profit {max(profit_forward, profit_reverse):.4f}")
                        return True
                except Exception as e:
                    logger.error(f"Arbitrage check failed: {e}")
        return False

    def backtest(self, periods=1000):
        equity = 100.0
        positions = {pair: 0.0 for pair in self.pairs}
        entry_prices = {pair: 0.0 for pair in self.pairs}
        for pair in self.pairs:
            df = self.get_ohlcv(self.primary, pair, '15m', periods)
            if df.empty:
                continue
            closes = df['close'].values
            volumes = df['volume'].values
            for i in range(RSI_PERIOD + 26, len(df)):
                slice_closes = closes[:i]
                slice_volumes = volumes[:i]
                rsi = calculate_rsi(slice_closes)
                macd, signal = calculate_macd(slice_closes)
                upper, middle, lower = calculate_bollinger_bands(slice_closes)
                vwap = calculate_vwap(slice_closes, slice_volumes)
                price = slice_closes[-1]
                avg_volume = slice_volumes.mean()
                # Mock model decision
                mock_prompt = f"RSI {rsi}, MACD {macd} signal {signal}, price {price} vs VWAP {vwap}, lower {lower} upper {upper}, vol spike {volumes[-1] > avg_volume*1.5}. Aggressive decision."
                decision, strong = random.choice([('BUY', True), ('SELL', False), ('HOLD', False)])  # Mock
                if decision == 'BUY' and equity > self.min_trade_amount:
                    risk = AGGRESSIVE_RISK if strong else RISK_PER_TRADE
                    buy_amt = (equity * risk) / price
                    positions[pair] += buy_amt * (1 - FEE)
                    equity -= buy_amt * price
                    entry_prices[pair] = price
                elif decision == 'SELL' and positions[pair] > 0:
                    sell_value = positions[pair] * price * (1 - FEE)
                    equity += sell_value
                    positions[pair] = 0.0
                # Stop-loss/take-profit
                if positions[pair] > 0:
                    unrealized = (price - entry_prices[pair]) / entry_prices[pair]
                    if unrealized <= -STOP_LOSS or unrealized >= TAKE_PROFIT:
                        sell_value = positions[pair] * price * (1 - FEE)
                        equity += sell_value
                        positions[pair] = 0.0
        total_equity = equity + sum(positions[p] * self.primary.fetch_ticker(p)['last'] for p in self.pairs)
        roi = (total_equity / 100.0 - 1) * 100
        logger.info(f"Backtest ROI: {roi:.2f}% over {periods} periods")
        return roi

    def generate_performance_report(self):
        if not os.path.exists('trades.csv'):
            return "No trade history available."
        trades = pd.read_csv('trades.csv')
        if trades.empty:
            return "No trades recorded."
        
        realized_pnl = 0.0
        unrealized_pnl = 0.0
        wins = 0
        losses = 0
        total_trades = 0
        open_positions = {}
        
        for pair in self.pairs:
            pair_trades = trades[trades['pair'] == pair]
            buys = pair_trades[pair_trades['type'] == 'buy']
            sells = pair_trades[pair_trades['type'] == 'sell']
            
            # Realized: Match buys and sells
            min_len = min(len(buys), len(sells))
            if min_len > 0:
                pnl = (sells['amount'].iloc[:min_len] * sells['price'].iloc[:min_len]) - (buys['amount'].iloc[:min_len] * buys['price'].iloc[:min_len])
                realized_pnl += pnl.sum()
                wins += (pnl > 0).sum()
                losses += (pnl < 0).sum()
                total_trades += min_len
            
            # Open positions: Excess buys
            if len(buys) > len(sells):
                open_amount = buys['amount'].iloc[len(sells):].sum()
                entry_price = buys['price'].iloc[-1]  # Last buy as approx entry
                current_price = self.primary.fetch_ticker(pair)['last']
                unrealized_pnl += (current_price - entry_price) * open_amount
                open_positions[pair] = open_amount
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        profit_factor = realized_pnl / abs(realized_pnl) if realized_pnl < 0 else realized_pnl / (abs(realized_pnl) if losses > 0 else 1)  # Simplified
        avg_return = (realized_pnl / total_trades) if total_trades > 0 else 0.0
        
        report = f"Performance: Realized PnL: {realized_pnl:.2f} EUR, Unrealized PnL: {unrealized_pnl:.2f} EUR, Win Rate: {win_rate:.2f}%, Avg Return/Trade: {avg_return:.2f} EUR, Profit Factor: {profit_factor:.2f}, Open Positions: {len(open_positions)}"
        logger.info(report)
        return report

    def trade(self, simulate=False):
        self.check_arbitrage(TRIANGLES)  # More frequent
        balance = self.primary.fetch_balance()
   

        eur = balance.get('EUR', {}).get('free', 0)
        btc_bal = balance.get('BTC', {}).get('free', 0)
        btc_price = self.primary.fetch_ticker(self.btc_pair)['last']
        
        
        usdc_bal = balance.get('USDC', {}).get('free', 0)
        usdc_price = self.primary.fetch_ticker(self.usdc_pair)['last']

        logger.info(f"Bitvavo Balance:\n EUR {eur:.2f},\n BTC {btc_bal:.6f} ({btc_price:.2f} EUR),\n USDC {usdc_bal:.2f} ({usdc_price:.2f} EUR)")

        total_value = eur + (btc_bal * btc_price) + (usdc_bal * usdc_price)
        positions = {}
        for pair in self.pairs:
            asset = pair.split('/')[0]
            pos = balance.get(asset, {}).get('free', 0)
            positions[pair] = pos
            price = self.primary.fetch_ticker(pair)['last']
            total_value += pos * price
        logger.info(f"Portfolio value: {total_value:.2f} EUR")

        articles = fetch_latest_news()
        sentiment = calculate_sentiment(articles) if articles else 0.0

        performance_report = self.generate_performance_report()
        logger.info(f"Performance Report: {performance_report}")

        for pair in self.pairs:
            df = self.get_ohlcv(self.primary, pair)
            if df.empty:
                continue
            closes = df['close'].values
            volumes = df['volume'].values
            rsi = calculate_rsi(closes)
            macd, signal = calculate_macd(closes)
            upper, middle, lower = calculate_bollinger_bands(closes)
            vwap = calculate_vwap(closes, volumes)
            avg_volume = volumes.mean()
            current_volume = volumes[-1]
            price = closes[-1]
            position = positions[pair]
            unrealized = (price - self.entry_prices[pair]) / self.entry_prices[pair] if position > 0 else 0
            logger.info(f"{pair} - Sentiment: {sentiment:.3f} - RSI: {rsi:.2f}, MACD: {macd:.8f}, Signal: {signal:.8f}, VWAP: {vwap:.8f}, Volume: {current_volume:.2f} (avg {avg_volume:.2f}), Unrealized PnL: {unrealized:.2f}")

            # Model decision with performance report
            prompt = f"""You are a cryptocurrency trading expert specializing in maximizing EUR cashflow through data-driven, adaptive trading strategies. 
            Your goal is to make decisions that balance risk and reward, learning from past performance to refine your approach.
            **Core Trading Rules:**
            - **Buy Conditions:** Trigger only when multiple indicators align for oversold conditions or price dips, AND market sentiment is positive. Require at least 2 confirming signals for entry.
            - **Sell Conditions:** Trigger when indicators show overbought conditions AND the current position is profitable (profit margin > 5%). Always prioritize taking profits to lock in gains.
            - **Hold Conditions:** Default action if no clear buy/sell signals, or if conditions are mixed. Override only with strong evidence from indicators or performance adaptation.

            **Indicator Thresholds (use these as guidelines, not absolutes):**
            - RSI: Buy if <30 (oversold), Sell if >75 (overbought).
            - MACD: Buy if MACD > Signal Line (bullish crossover), Sell if MACD < Signal Line (bearish crossover).
            - Bollinger Bands: Buy if Price < Lower Band (potential dip), Sell if Price > Upper Band (potential peak).
            - VWAP: Buy if Price < VWAP (undervalued), Sell if Price > VWAP (overvalued).
            - Volume: Confirm signals if Current Volume > 1.5 * Average Volume (spike indicates momentum).
            - Sentiment Score: Buy if >0.2 (positive), Sell if <-0.2 (negative), Hold if between -0.2 and 0.2 (neutral).

            **Current Profit Margin Integration:**
            - If >5%: Strongly consider Sell to realize profits.
            - If <0%: Consider Buy to accumulate on dips (average down), but only if indicators support.
            - Otherwise: Hold, unless overridden by strong indicator alignment.

            **Adaptive Learning from Past Performance:**
            Use the provided performance report to adjust your risk tolerance:
            - If overall PnL is negative or max drawdown >10%: Adopt a conservative stance—require 3+ confirming indicators, wider thresholds (e.g., RSI <25 for buy, >75 for sell), and avoid trades on moderate signals.
            - If win rate <50%: Tighten criteria—demand high-confidence signals (e.g., volume spike + sentiment alignment), and increase profit target to >7% for sells.
            - If performing well (win rate >60% and positive PnL): Be more aggressive—act on 2 confirming indicators, use standard thresholds, and consider scaling positions on strong signals.
            - Always prioritize capital preservation over aggressive growth.

            **Input Data:**
            - Pair: {pair}
            - Indicators: RSI {rsi}, MACD {macd}, Signal Line {signal}, Price {price}, Lower Band {lower}, Upper Band {upper}, VWAP {vwap},
              average volume {avg_volume}, current volume {current_volume}, Sentiment {sentiment}.
            - Past Performance: {performance_report}
            - Current Unrealized Profit Margin: {unrealized * 100:.2f}% (if >0, position is profitable; if <0, at a loss).

            Reason step-by-step internally:
            1. Evaluate indicators against thresholds for buy/sell/hold signals.
            2. Factor in current profit margin.
            3. Adjust based on performance report (conservative, tightened, or aggressive).
            4. Resolve conflicts by prioritizing sell for profits, buy for confirmed dips, hold for uncertainty.
            5. Assess confidence: If very strong alignment (3+ indicators + favorable adaptation), note internally as "strong" but output only the base action.

            Output format: Respond with ONLY the single-word decision in lowercase: "buy", "sell", or "hold". No explanations,
            no additional text, no variations like "strong buy". """
            decision, strong = self.query_model(prompt)
            
            logger.info(f"position {position}")
            if position > 0:
                unrealized = (price - self.entry_prices[pair]) / self.entry_prices[pair]
                if unrealized <= -STOP_LOSS or unrealized >= TAKE_PROFIT:
                    self._execute_sell(pair, position, price, simulate, reason=f"{'Stop-loss' if unrealized < 0 else 'Take-profit'} triggered")
                    continue

            if decision == 'BUY' and eur > self.min_trade_amount and sentiment > 0.2:
                risk = AGGRESSIVE_RISK if strong else RISK_PER_TRADE
                amount = self.primary.amount_to_precision(pair, (total_value * risk) / price)
                min_amt = self.primary.markets[pair]['limits']['amount']['min']
                if float(amount) >= min_amt:
                    self._execute_buy(pair, amount, price, simulate)
                else:
                    logger.warning(f"Buy amount {amount} below min {min_amt}")

            elif decision == 'SELL' and position > 0:
                self._execute_sell(pair, position, price, simulate, reason="Model sell")

        # BTC logic (kept, but aggressive on dips)
        if eur > MIN_EUR_FOR_BTC:
            btc_df = self.get_ohlcv(self.primary, self.btc_pair)
            if not btc_df.empty:
                btc_closes = btc_df['close'].values
                pct_change = (btc_closes[-1] - btc_closes[-5]) / btc_closes[-5] if len(btc_closes) > 5 else 0
                if pct_change < -BTC_DIP or sentiment < -0.2:
                    buy_amt = self.primary.amount_to_precision(self.btc_pair, (eur * 0.1) / btc_closes[-1])
                    self._execute_buy(self.btc_pair, buy_amt, btc_closes[-1], simulate, is_btc=True)

    def _execute_buy(self, pair, amount, price, simulate, is_btc=False):
        if simulate:
            logger.info(f"Simulated BUY {pair}: {amount} at {price}")
            return
        try:
            order = self.primary.create_market_buy_order(pair, amount)
            logger.info(f"BUY executed: {order}")
            self.entry_prices[pair] = price
            self._log_trade(time.time(), 'buy', amount, price, pair)
        except Exception as e:
            logger.error(f"BUY failed for {pair}: {e}")

    def _execute_sell(self, pair, amount, price, simulate, reason=""):
        if simulate:
            logger.info(f"Simulated SELL {pair}: {amount} at {price} ({reason})")
            return
        try:
            order = self.primary.create_market_sell_order(pair, amount)
            logger.info(f"SELL executed: {order} ({reason})")
            self._log_trade(time.time(), 'sell', amount, price, pair)
        except Exception as e:
            logger.error(f"SELL failed for {pair}: {e}")

    def _log_trade(self, timestamp, trade_type, amount, price, pair):
        df = pd.DataFrame({'time': [timestamp], 'type': [trade_type], 'pair': [pair], 'amount': [float(amount)], 'price': [price]})
        header = not os.path.exists('trades.csv')
        try:
            df.to_csv('trades.csv', mode='a', header=header, index=False)
        except Exception as e:
            logger.error(f"Failed to log trade to CSV: {e}")

def main(simulate=False):
    try:
        import pandas
    except ImportError:
        logger.error("Pandas not installed; please run: pip install pandas")
        return
    bot = TradingBot(bitvavo, kraken, 'BTC/EUR', 'USDC/EUR', min_trade_amount=10, max_loss=0.05)
    bot.backtest()
    trade_count = 0
    while True:
        bot.trade(simulate)
        trade_count += 1
        if trade_count % 10 == 0 and os.path.exists('trades.csv'):
            try:
                trades = pd.read_csv('trades.csv')
                profits = []
                for pair in PAIRS:
                    buys = trades[(trades['type'] == 'buy') & (trades['pair'] == pair)]
                    sells = trades[(trades['type'] == 'sell') & (trades['pair'] == pair)]
                    if len(buys) > 0 and len(sells) > 0:
                        pair_profits = sells['amount'] * sells['price'] - buys['amount'].iloc[:len(sells)] * buys['price'].iloc[:len(sells)]
                        profits.extend(pair_profits)
                if profits:
                    win_rate = (np.array(profits) > 0).mean()
                    if win_rate < 0.5:
                        bot.rsi_buy = min(bot.rsi_buy + 5, 50)
                        logger.info(f"Adjusted RSI_BUY to {bot.rsi_buy} (win rate {win_rate:.2f})")
            except Exception as e:
                logger.error(f"Optimization failed: {e}")

        current_time = time.time()
        next_run = ((current_time // 900) + 1) * 900
        sleep_time = next_run - current_time
        logger.info(f"Sleeping for {sleep_time:.2f} seconds until {time.ctime(next_run)}")
        time.sleep(sleep_time)
        if simulate:
            logger.info("Simulation mode active; exiting after one run.")
            break

if __name__ == "__main__":
    main(simulate=False)