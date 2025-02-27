import asyncio

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
import pytz
from datetime import datetime, timedelta
import ta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import requests
import yfinance as yf

from alerts import send_telegram_alert
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
options_stocks = ['TSLA', 'PLTR', 'NVDA', 'AAPL', 'AMZN', 'ASML', 'AVGO', 'COIN', 'GOOG', 'META', 'MSTR', 'NFLX',
                      'ORCL', 'QDTE', 'SPY', 'SMCI', 'UNH','QQQ','LRCX','JPM','SNOW','FI','BA','MELI','INTC','AMD','MARA','MSFT']

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

def get_historical_data(symbol, timeframe='15Min', limit=60):
    """Get historical data for a symbol using Alpaca's API."""
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            end = datetime.now(pytz.UTC)
            start = end - timedelta(days=limit * 2)
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            print(f"Fetching historical data for {symbol} from {start_str} to {end_str}...")
            bars = api.get_bars(
                symbol,
                timeframe,
                start=start_str,
                end=end_str,
                adjustment='raw',
                feed='iex'
            ).df
            if len(bars) < limit:
                print(f"Warning: Only got {len(bars)} bars for {symbol}")
                return None
            return bars
        except Exception as e:
            print(e)
            retry_count += 1
            if retry_count < MAX_RETRIES:
                print(f"Retry {retry_count} for {symbol}")
                time.sleep(2 ** retry_count)
            else:
                print(f"Failed to get data for {symbol} after {MAX_RETRIES} retries")
                return None


def calculate_technical_indicators(df):
    """Calculate technical indicators with improved parameters and additional confirmations."""
    # Volume analysis
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Price action
    df['high_low_diff'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']

    # Trend Detection
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma200'] = df['close'].rolling(window=200).mean()

    # Enhanced MACD (longer periods for more reliable signals)
    macd_indicator = MACD(df['close'],
                          window_slow=26,  # Standard period
                          window_fast=12,  # Standard period
                          window_sign=9)  # Standard period
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()

    # RSI with standard period
    rsi_indicator = RSIIndicator(df['close'], window=14)  # Standard period
    df['rsi'] = rsi_indicator.rsi()

    # Enhanced Bollinger Bands
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # ATR for volatility measurement
    df['tr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # Momentum
    df['roc'] = ta.momentum.roc(df['close'], window=10)

    return df


def analyze_market_conditions(df, options_data=None):
    """Enhanced market analysis with multiple confirmation signals."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    analysis = {
        'entry_signals': [],
        'exit_signals': [],
        'risk_level': 0,
        'trend_strength': 0,
        'volatility_state': ''
    }

    # Trend Analysis
    trend_score = 0

    # Multiple timeframe analysis
    if latest['close'] > latest['sma20'] > latest['sma50']:
        trend_score += 2
        print("Strong uptrend confirmed by multiple SMAs (+2)")
    elif latest['close'] < latest['sma20'] < latest['sma50']:
        trend_score -= 2
        print("Strong downtrend confirmed by multiple SMAs (-2)")

    # Volume confirmation
    if latest['volume_ratio'] > 1.5:
        trend_score += 1
        print("Strong volume confirmation (+1)")

    # Price action confirmation
    if latest['close_open_diff'] > 0 and latest['high_low_diff'] > df['high_low_diff'].mean():
        trend_score += 1
        print("Strong bullish price action (+1)")
    elif latest['close_open_diff'] < 0 and latest['high_low_diff'] > df['high_low_diff'].mean():
        trend_score -= 1
        print("Strong bearish price action (-1)")

    analysis['trend_strength'] = trend_score

    # Momentum Analysis
    momentum_score = 0

    # MACD Analysis with confirmation
    if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > prev['macd_hist']:
        momentum_score += 2
        print("Strong MACD bullish confirmation (+2)")
    elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < prev['macd_hist']:
        momentum_score -= 2
        print("Strong MACD bearish confirmation (-2)")

    # RSI Analysis with dynamic thresholds
    if 40 <= latest['rsi'] <= 60:  # Neutral zone
        if latest['rsi'] > prev['rsi']:  # Rising RSI
            momentum_score += 1
            print("Rising RSI in neutral zone (+1)")
        else:  # Falling RSI
            momentum_score -= 1
            print("Falling RSI in neutral zone (-1)")

    # Rate of Change confirmation
    if latest['roc'] > 0 and latest['roc'] > df['roc'].mean():
        momentum_score += 1
        print("Strong positive momentum (+1)")
    elif latest['roc'] < 0 and latest['roc'] < df['roc'].mean():
        momentum_score -= 1
        print("Strong negative momentum (-1)")

    # Volatility Analysis
    volatility_score = 0
    current_atr = latest['tr']
    avg_atr = df['tr'].mean()

    if current_atr < avg_atr * 0.8:
        volatility_state = 'low'
        volatility_score += 1
        print("Low volatility environment (+1)")
    elif current_atr > avg_atr * 1.2:
        volatility_state = 'high'
        volatility_score -= 1
        print("High volatility environment (-1)")
    else:
        volatility_state = 'normal'

    analysis['volatility_state'] = volatility_state

    # Risk Assessment
    risk_level = 0

    # Market structure risk
    if latest['close'] > latest['bb_upper']:
        risk_level -= 1
        print("Overbought conditions (-1)")
    elif latest['close'] < latest['bb_lower']:
        risk_level -= 1
        print("Oversold conditions (-1)")

    # Options market risk
    if options_data:
        if options_data.get('iv_percentile', 50) < 30:
            risk_level += 1
            print("Low IV environment (+1)")
        elif options_data.get('iv_percentile', 50) > 70:
            risk_level -= 1
            print("High IV environment (-1)")

        if options_data.get('put_call_ratio', 1) < 0.7:
            risk_level += 1
            print("Bullish options sentiment (+1)")
        elif options_data.get('put_call_ratio', 1) > 1.3:
            risk_level -= 1
            print("Bearish options sentiment (-1)")

    analysis['risk_level'] = risk_level

    # Entry Signals (requiring multiple confirmations)
    if (trend_score >= 2 and momentum_score >= 2 and risk_level >= 1 and
            volatility_state != 'high'):
        analysis['entry_signals'].append('strong_bullish')
        print("Entry Signal: strong_bullish")
    elif (trend_score <= -2 and momentum_score <= -2 and risk_level >= 1 and
          volatility_state != 'high'):
        analysis['entry_signals'].append('strong_bearish')
        print("Entry Signal: strong_bearish")

    # Exit Signals
    if latest['rsi'] > 75 or latest['rsi'] < 25:
        analysis['exit_signals'].append('rsi_extreme')
        print("Exit Signal: RSI extreme")
    if (latest['close'] < latest['sma20'] and prev['close'] > prev['sma20'] and
            trend_score > 0):
        analysis['exit_signals'].append('trend_reversal')
        print("Exit Signal: Trend reversal")
    if volatility_state == 'high' and abs(latest['roc']) > df['roc'].std() * 2:
        analysis['exit_signals'].append('volatility_spike')
        print("Exit Signal: Volatility spike")

    return analysis


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
    """Fetch options data using yfinance with rate limiting and improved error handling."""
    try:
        # Add delay between requests
        time.sleep(60 / CALLS_PER_MINUTE)  # Ensure even distribution of requests

        stock = yf.Ticker(symbol)
        if not stock.options:
            print(f"No available option expirations for {symbol}")
            return None

        # Select the first (nearest) expiration
        expiry = stock.options[0]

        # Add error handling for option chain retrieval
        try:
            options_chain = stock.option_chain(expiry)
        except Exception as e:
            print(f"Error fetching option chain for {symbol}: {e}")
            return None

        # Get the current underlying price with fallback options
        try:
            underlying_price = stock.info.get("regularMarketPrice")
            if underlying_price is None:
                # First fallback: try last quote
                quote = stock.history(period="1d")
                if not quote.empty:
                    underlying_price = quote['Close'].iloc[-1]
                else:
                    print(f"Unable to get price data for {symbol}")
                    return None
        except Exception as e:
            print(f"Error fetching price data for {symbol}: {e}")
            return None

        # Calculate the nearest strike price from the calls DataFrame
        calls_df = options_chain.calls
        if calls_df.empty:
            print(f"No calls data available for {symbol}")
            return None

        # Compute the absolute difference between each strike and the underlying price
        diff = (calls_df['strike'] - underlying_price).abs()
        nearest_strike = calls_df.loc[diff.idxmin(), 'strike']

        # Calculate implied volatility percentile with error handling
        try:
            call_ivs = options_chain.calls['impliedVolatility'].dropna().values
            put_ivs = options_chain.puts['impliedVolatility'].dropna().values
            iv_values = np.concatenate([call_ivs, put_ivs])
            iv_percentile = float(np.percentile(iv_values, percentile)) if len(iv_values) > 0 else None
        except Exception as e:
            print(f"Error calculating IV percentile for {symbol}: {e}")
            iv_percentile = None

        # Calculate put/call ratio with error handling
        try:
            total_call_volume = options_chain.calls['volume'].sum()
            total_put_volume = options_chain.puts['volume'].sum()
            put_call_ratio = float(total_put_volume / total_call_volume) if total_call_volume > 0 else None
        except Exception as e:
            print(f"Error calculating put/call ratio for {symbol}: {e}")
            put_call_ratio = None

        return {
            'iv_percentile': iv_percentile,
            'put_call_ratio': put_call_ratio,
            'nearest_strike': nearest_strike,
            'underlying_price': underlying_price
        }
    except Exception as e:
        print(f"Error fetching options data for {symbol}: {e}")
        # Add exponential backoff retry
        retry_delay = 60  # Start with 1 minute delay
        max_retries = 3
        for retry in range(max_retries):
            print(f"Retrying in {retry_delay} seconds... (Attempt {retry + 1}/{max_retries})")
            time.sleep(retry_delay)
            try:
                return get_options_data(symbol, percentile)
            except Exception as retry_e:
                print(f"Retry {retry + 1} failed: {retry_e}")
                retry_delay *= 2  # Exponential backoff
        return None


def get_filtered_stocks(top_n=10):
    """Filter stocks for directional options trades with improved rate limiting."""
    filtered_stocks = []

    # Add delay between stock analysis
    delay_between_stocks = 5  # seconds

    print("\nAnalyzing stocks for directional options trades...")

    for stock in options_stocks:
        try:
            print(f"\nAnalyzing {stock}...")

            # Get historical data with retry mechanism
            bars = None
            retries = 3
            for attempt in range(retries):
                try:
                    bars = get_historical_data(stock)
                    if bars is not None:
                        break
                except Exception as e:
                    print(f"Attempt {attempt + 1}/{retries} failed: {e}")
                    time.sleep(delay_between_stocks * (attempt + 1))

            if bars is None:
                print(f"Failed to get historical data for {stock} after {retries} attempts")
                continue

            df = calculate_technical_indicators(bars)

            # Add delay before fetching options data
            time.sleep(delay_between_stocks)

            options_data = get_options_data(stock)
            if not options_data:
                print(f"❌ {stock}: No options data available")
                continue

            analysis = analyze_market_conditions(df, options_data)

            # Determine trade direction with more stringent criteria
            trade_decision = None
            if 'strong_bullish' in analysis['entry_signals'] and analysis['risk_level'] >= 1:
                trade_decision = 'CALL'
            elif 'strong_bearish' in analysis['entry_signals'] and analysis['risk_level'] >= 1:
                trade_decision = 'PUT'

            if trade_decision:
                trade_info = {
                    'symbol': stock,
                    'price': options_data['nearest_strike'],
                    'direction': trade_decision,
                    'nearest_strike': options_data['nearest_strike'],
                    'underlying_price': options_data['underlying_price']
                }
                filtered_stocks.append(trade_info)
                print(
                    f"✅ {stock}: {trade_decision} opportunity at ${options_data['underlying_price']:.2f} nearest strike at ${options_data['nearest_strike']:.2f}")
            else:
                print(f"❌ {stock}: No clear directional opportunity")

            # Add delay between stocks
            time.sleep(delay_between_stocks)

        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
            continue

    print(f"\nFound {len(filtered_stocks)} potential trades:")
    for trade in filtered_stocks:
        message = f"{trade['symbol']}: {trade['direction']} at ${trade['price']:.2f}"
        print(message)
        asyncio.run(send_telegram_alert(message))

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
