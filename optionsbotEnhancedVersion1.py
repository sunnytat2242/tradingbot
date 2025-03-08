import asyncio

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
import pytz
from datetime import datetime, timedelta
import ta
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import requests
import yfinance as yf

from alerts import send_telegram_alert
from analysis import analyze_market_conditions_day_trading
from config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPHA_VANTAGE_API_KEY
import time
from ratelimit import sleep_and_retry, limits

# Configure rate limits (yfinance typically allows 2000 requests per hour)
CALLS_PER_HOUR = 1800  # Setting slightly below limit to be safe
PERIOD_IN_SECONDS = 3600
CALLS_PER_MINUTE = CALLS_PER_HOUR / 60
# Alpaca Configuration
BASE_URL = 'https://paper-api.alpaca.markets'
MARKET_URL = 'https://data.alpaca.markets'

# Initialize client
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL, api_version='v2')

# Trading Parameters
RISK_PER_TRADE = 0.02
MIN_STOCK_PRICE = 20.0
MAX_STOCKS_TO_SCAN = 100
API_TIMEOUT = 10
MAX_RETRIES = 10
options_stocks = ['TSLA', 'PLTR']
#, 'NVDA', 'AAPL', 'AMZN', 'ASML', 'AVGO', 'COIN', 'GOOG', 'META', 'MSTR', 'NFLX'
 #                     ,'ORCL', 'QDTE', 'SPY', 'SMCI', 'UNH','QQQ','LRCX','JPM','SNOW','FI','BA','MELI','INTC','AMD','MARA','MSFT']

def get_stock_data(symbols):
    stock_data = []
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_API_SECRET
    }
    for symbol in symbols:
        endpoint = f"{MARKET_URL}/v2/stocks/{symbol}/quotes/latest"
        response = requests.get(endpoint, headers=headers)
        data = response.json()

        # Extracting the latest price and volume
        try:
            price = data["quote"]["ap"]  # Ask price
            volume = data["quote"]["as"]
            stock_data.append({"symbol": symbol, "price": price, "volume": volume})
        except KeyError:
            stock_data.append({"symbol": symbol, "error": "Data unavailable"})

        time.sleep(1)  # To avoid API rate limits

    return stock_data


def get_historical_data(symbol, timeframe='5Min', limit=50):
    """Get historical data for a symbol using Alpaca's API."""
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            end = datetime.now(pytz.UTC)
            start = end - timedelta(days=5)
            while start.weekday() >= 5:  # Avoid weekends
                start -= timedelta(days=1)
            start_str, end_str = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
            print(f"Fetching historical data for {symbol} from {start_str} to {end_str}...")
            bars = api.get_bars(symbol, timeframe, start=start_str, end=end_str, adjustment='raw', feed='iex').df

            if len(bars) < limit:
                print(f"Warning: Only got {len(bars)} bars for {symbol}")
                return None
            return bars
        except Exception as e:
            print(e)
            retry_count += 1
            time.sleep(2 ** retry_count)
    print(f"Failed to get data for {symbol} after {MAX_RETRIES} retries")
    return None


def calculate_technical_indicators(df):
    """Calculate technical indicators with faster responsiveness."""
    if df is None or df.empty:
        print("Error: DataFrame is empty.")
        return None

    df = df.copy()
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['high_low_diff'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']

    for window in [5, 10, 20, 50, 200]:
        df[f'sma{window}'] = ta.trend.SMAIndicator(df['close'], window=window).sma_indicator()

    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=1).volume_weighted_average_price()

    macd_indicator = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()  # Faster RSI

    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'], df['bb_lower'], df['bb_mid'] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()

    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    df['roc'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()

    df.dropna(subset=['sma5', 'sma10', 'macd', 'rsi', 'atr', 'vwap'], inplace=True)
    return df


def get_position_size(analysis, account_size, stock_price):
    """Calculate position size with enhanced risk management."""
    base_risk = RISK_PER_TRADE

    # Risk multiplier based on market conditions
    if analysis['risk_level'] >= 2 and analysis['trend_strength'] >= 2:
        risk_multiplier = 1.0
    elif analysis['risk_level'] >= 1 and analysis['trend_strength'] >= 1:
        risk_multiplier = 0.75
    else:
        risk_multiplier = 0.5

    # Additional volatility adjustment
    if analysis['volatility_state'] == 'high':
        risk_multiplier *= 0.6
    elif analysis['volatility_state'] == 'normal':
        risk_multiplier *= 0.8

    # Calculate final position size
    risk_amount = account_size * base_risk * risk_multiplier
    position_size = risk_amount / stock_price

    # Ensure minimum position size
    min_position = 100  # Minimum position value in dollars
    if position_size * stock_price < min_position:
        return 0

    return np.floor(position_size)


@sleep_and_retry
@limits(calls=CALLS_PER_HOUR, period=PERIOD_IN_SECONDS)
def get_options_data(symbol, percentile=50):
    CALLS_PER_MINUTE = 30
    
    try:
        time.sleep(60 / CALLS_PER_MINUTE)
        stock = yf.Ticker(symbol)
        if not stock.options:
            print(f"No available option expirations for {symbol}")
            return None

        expiry = stock.options[0]
        options_chain = stock.option_chain(expiry)

        underlying_price = stock.info.get("regularMarketPrice")
        if underlying_price is None:
            quote = stock.history(period="1d")
            if not quote.empty:
                underlying_price = quote['Close'].iloc[-1]
            else:
                print(f"Unable to get price data for {symbol}")
                return None

        calls_df = options_chain.calls
        if calls_df.empty:
            print(f"No calls data available for {symbol}")
            return None

        diff = (calls_df['strike'] - underlying_price).abs()
        nearest_strike = calls_df.loc[diff.idxmin(), 'strike']

        call_ivs = options_chain.calls['impliedVolatility'].dropna().values
        put_ivs = options_chain.puts['impliedVolatility'].dropna().values
        current_ivs = np.concatenate([call_ivs, put_ivs])
        iv_percentile = float(np.percentile(current_ivs, percentile) / np.max(current_ivs) * 100 if len(current_ivs) > 0 else 50.0)  # Normalize to 0-100

        iv_hist = []
        hist = stock.history(period="30d")
        for exp in stock.options[:min(3, len(stock.options))]:
            chain = stock.option_chain(exp)
            iv_hist.extend(chain.calls['impliedVolatility'].dropna().values)
            iv_hist.extend(chain.puts['impliedVolatility'].dropna().values)
        iv_rank = float(np.percentile(current_ivs, percentile) / np.percentile(iv_hist, 100) * 100 if iv_hist else 50.0)
        iv_rank = min(max(iv_rank, 0), 100)

        total_call_volume = options_chain.calls['volume'].sum()
        total_put_volume = options_chain.puts['volume'].sum()
        put_call_ratio = float(total_put_volume / total_call_volume) if total_call_volume > 0 else 1.0

        print(f"{symbol} Options: IV Percentile={iv_percentile:.1f}, IV Rank={iv_rank:.1f}, Put/Call={put_call_ratio:.2f}")
        return {
            'iv_percentile': iv_percentile,
            'iv_rank': iv_rank,
            'put_call_ratio': put_call_ratio,
            'nearest_strike': nearest_strike,
            'underlying_price': underlying_price
        }
    except Exception as e:
        print(f"Error fetching options data for {symbol}: {e}")
        max_retries = 3
        retry_delay = 60
        for retry in range(max_retries):
            time.sleep(retry_delay)
            try:
                return get_options_data(symbol, percentile)
            except Exception as retry_e:
                print(f"Retry {retry + 1} failed: {retry_e}")
                retry_delay *= 2
        return None


def get_filtered_stocks(top_n=10):
    filtered_stocks = []
    delay_between_stocks = 1

    print("\nAnalyzing stocks for directional options trades...")

    for stock in options_stocks:
        try:
            print(f"\nAnalyzing {stock}...")
            bars = get_historical_data(stock)
            if bars is None:
                print(f"Failed to get historical data for {stock}")
                continue

            df = calculate_technical_indicators(bars)
            if df is None:
                continue

            time.sleep(delay_between_stocks)

            options_data = get_options_data(stock)
            if not options_data:
                print(f"❌ {stock}: No options data available")
                continue

            analysis = analyze_market_conditions_day_trading(df, options_data)
            print(f"Risk level for {stock}: {analysis['risk_level']}")
            print(f"Entry signals for {stock}: {analysis['entry_signals']}")
            print(f"Exit signals for {stock}: {analysis['exit_signals']}")

            trade_decision = None
            if 'strong_bullish' in analysis['entry_signals'] and analysis['risk_level'] >= 1 and not analysis['exit_signals']:
                trade_decision = 'CALL'
            elif 'strong_bearish' in analysis['entry_signals'] and analysis['risk_level'] >= 1 and not analysis['exit_signals']:
                trade_decision = 'PUT'

            if trade_decision:
                trade_info = {
                    'symbol': stock,
                    'price': options_data['underlying_price'],
                    'direction': trade_decision,
                    'nearest_strike': options_data['nearest_strike'],
                    'underlying_price': options_data['underlying_price'],
                    'trend_strength': analysis['trend_strength'],
                    'momentum_score': sum([1 if 'bullish' in s else -1 if 'bearish' in s else 0 for s in analysis['entry_signals']])
                }
                filtered_stocks.append(trade_info)
                message = f"✅ {stock}: {trade_decision} opportunity at ${options_data['underlying_price']:.2f}, nearest strike ${options_data['nearest_strike']:.2f}"
                asyncio.run(send_telegram_alert(message))
            else:
                print(f"❌ {stock}: No clear directional opportunity")

            time.sleep(delay_between_stocks)

        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
            continue

    filtered_stocks = sorted(filtered_stocks, key=lambda x: abs(x['trend_strength']) + abs(x['momentum_score']), reverse=True)[:top_n]

    print(f"\nFound {len(filtered_stocks)} potential trades (top {top_n}):")
    for trade in filtered_stocks:
        message = f"{trade['symbol']}: {trade['direction']} at ${trade['underlying_price']:.2f}, strike ${trade['nearest_strike']:.2f}"
        print(message)

    return filtered_stocks

def options_trading_bot():
    """Main options trading loop with refined logic."""
    while True:
        try:
            print("\n=== Scanning for Options Trading Opportunities ===")
            filtered_stocks = get_filtered_stocks(MAX_STOCKS_TO_SCAN)
            if not filtered_stocks:
                print("No stocks meet criteria. Retrying in 5 minutes...")
                time.sleep(300)
                continue
            for stock in filtered_stocks:
                symbol = stock['symbol']
                print(f"\nChecking options for {symbol}...")
                options_data = get_options_data(symbol)
                if options_data is None:
                    continue
                # Simple trading rules based on IV and put/call ratio
                if (options_data.get('iv_percentile') is not None and options_data['iv_percentile'] > 50 and
                        options_data.get('put_call_ratio') is not None and options_data['put_call_ratio'] > 1.0):
                    print(f"⚠️ High IV & Bearish Sentiment: Consider selling premium on {symbol}")
                elif (options_data.get('iv_percentile') is not None and options_data['iv_percentile'] < 20 and
                      options_data.get('put_call_ratio') is not None and options_data['put_call_ratio'] < 0.7):
                    print(f"🚀 Low IV & Bullish Sentiment: Consider buying calls on {symbol}")
                time.sleep(1)
            print("\nScan complete. Waiting for next scan...")
            time.sleep(300)
        except KeyboardInterrupt:
            print("\nStopping options bot...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(60)


if __name__ == '__main__':
    options_trading_bot()
