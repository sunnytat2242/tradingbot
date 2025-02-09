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
from config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPHA_VANTAGE_API_KEY

# Alpaca Configuration
BASE_URL = 'https://paper-api.alpaca.markets'
MARKET_URL = 'https://data.alpaca.markets'

# Initialize client
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL, api_version='v2')

# Trading Parameters
RISK_PER_TRADE = 0.02
MIN_STOCK_PRICE = 20.0
MAX_STOCKS_TO_SCAN = 10
API_TIMEOUT = 10
MAX_RETRIES = 3


def get_top_active_stocks_alpha(top_n=10):
    """Fetch the top N most active stocks by volume using Alpha Vantage API."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TOP_GAINERS_LOSERS",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "most_actively_traded" not in data:
            print(f"Unexpected response format: {data}")
            return []
        active_stocks = []
        for stock in data["most_actively_traded"][:top_n]:
            symbol = stock.get("ticker")
            price = float(stock.get("price", 0))
            volume = int(stock.get("volume", 0))
            active_stocks.append({
                "symbol": symbol,
                "price": price,
                "volume": volume
            })
        return active_stocks
    except requests.RequestException as e:
        print(f"Error fetching most active stocks: {e}")
        return []


def get_historical_data(symbol, timeframe='1D', limit=200):
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
    """Calculate technical indicators using a refined set of signals."""
    # MACD Calculation
    macd_indicator = MACD(df['close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()

    # RSI Calculation
    rsi_indicator = RSIIndicator(df['close'])
    df['rsi'] = rsi_indicator.rsi()

    # Bollinger Bands (20-period, 2 standard deviations)
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # Trend indicator: using EMA9 and SMA20
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['sma20'] = df['close'].rolling(window=20).mean()

    # Hull Moving Average (optional, for faster response)
    def hull_moving_average(series, period):
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        wma1 = series.rolling(window=half_period).mean()
        wma2 = series.rolling(window=period).mean()
        diff = 2 * wma1 - wma2
        return diff.rolling(window=sqrt_period).mean()

    df['hma20'] = hull_moving_average(df['close'], 20)

    print("Technical indicators calculated.")
    return df


def analyze_market_conditions(df, options_data=None):
    """Analyze market conditions with refined thresholds and aggregated signals."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    analysis = {
        'entry_signals': [],
        'exit_signals': [],
        'risk_level': 0,
        'trend_strength': 0,
        'volatility_state': ''
    }

    # Trend Analysis using EMA9 vs SMA20 and HMA20 confirmation
    trend_score = 0
    if latest['ema9'] > latest['sma20']:
        trend_score += 1
        print("EMA9 > SMA20: Bullish trend (+1)")
    else:
        trend_score -= 1
        print("EMA9 <= SMA20: Bearish trend (-1)")
    if latest['close'] > latest['hma20']:
        trend_score += 1
        print("Close > HMA20: Bullish momentum (+1)")
    else:
        trend_score -= 1
        print("Close <= HMA20: Bearish momentum (-1)")
    analysis['trend_strength'] = trend_score

    # Momentum Analysis using MACD and RSI (using a neutral RSI range 40-60)
    momentum_score = 0
    if latest['macd'] > latest['macd_signal']:
        momentum_score += 1
        print("MACD > Signal: Bullish momentum (+1)")
    else:
        momentum_score -= 1
        print("MACD <= Signal: Bearish momentum (-1)")
    if 40 < latest['rsi'] < 60:
        momentum_score += 1
        print("RSI in neutral zone (40-60): Positive momentum (+1)")
    else:
        momentum_score -= 1
        print("RSI outside neutral zone: Negative momentum (-1)")

    # Volatility Analysis using Bollinger Band width
    if latest['bb_width'] > df['bb_width'].mean() * 1.2:
        volatility_state = 'high'
        print("High volatility based on Bollinger width")
    else:
        volatility_state = 'low'
        print("Low volatility based on Bollinger width")
    analysis['volatility_state'] = volatility_state

    # Risk level based on RSI, volatility, and options data (IV percentile)
    risk_level = 0
    if 40 < latest['rsi'] < 60:
        risk_level += 1
    if volatility_state == 'low':
        risk_level += 1
    if options_data and options_data.get('iv_percentile') is not None:
        if options_data['iv_percentile'] < 30:
            risk_level += 1
            print("Low IV environment: risk level increased (+1)")
        elif options_data['iv_percentile'] > 70:
            risk_level -= 1
            print("High IV environment: risk level decreased (-1)")
    analysis['risk_level'] = risk_level

    # Determine entry signals based on aggregated scores
    if trend_score >= 1 and momentum_score >= 1 and risk_level >= 2:
        analysis['entry_signals'].append('strong_bullish')
        print("Entry Signal: strong_bullish")
    elif trend_score <= -1 and momentum_score <= -1 and risk_level >= 2:
        analysis['entry_signals'].append('strong_bearish')
        print("Entry Signal: strong_bearish")

    # Exit Conditions: if RSI is extreme or a crossover happens
    if latest['rsi'] > 70 or latest['rsi'] < 30:
        analysis['exit_signals'].append('rsi_extreme')
        print("Exit Signal: RSI extreme")
    if latest['close'] < latest['sma20'] and prev['close'] > prev['sma20']:
        analysis['exit_signals'].append('ma_crossover')
        print("Exit Signal: MA crossover")

    print("Market conditions analyzed.")
    return analysis


def get_position_size(analysis, account_size, stock_price):
    """Calculate position size based on dynamic risk management."""
    base_risk = RISK_PER_TRADE  # 2% base risk per trade
    if analysis['risk_level'] >= 3:
        risk_multiplier = 1.0
    elif analysis['risk_level'] == 2:
        risk_multiplier = 0.75
    else:
        risk_multiplier = 0.5
    if analysis['volatility_state'] == 'high':
        risk_multiplier *= 0.8
    risk_amount = account_size * base_risk * risk_multiplier
    position_size = risk_amount / stock_price
    return np.floor(position_size)


def get_options_data(symbol):
    """Fetch options data using yfinance and calculate IV percentile and put/call ratio."""
    try:
        stock = yf.Ticker(symbol)
        if not stock.options:
            print(f"No available option expirations for {symbol}")
            return None
        expiry = stock.options[0]
        options_chain = stock.option_chain(expiry)
        call_ivs = options_chain.calls['impliedVolatility'].dropna().values
        put_ivs = options_chain.puts['impliedVolatility'].dropna().values
        iv_values = np.concatenate([call_ivs, put_ivs])
        iv_percentile = float(np.percentile(iv_values, 50)) if len(iv_values) > 0 else None
        total_call_volume = options_chain.calls['volume'].sum()
        total_put_volume = options_chain.puts['volume'].sum()
        put_call_ratio = float(total_put_volume / total_call_volume) if total_call_volume > 0 else None
        return {'iv_percentile': iv_percentile, 'put_call_ratio': put_call_ratio}
    except Exception as e:
        print(f"Error fetching options data for {symbol}: {e}")
        return None


def get_filtered_stocks(top_n=10):
    """Filter stocks for directional options trades using refined technical analysis."""
    active_stocks = get_top_active_stocks_alpha(top_n)
    filtered_stocks = []
    print("\nAnalyzing stocks for directional options trades...")
    for stock in active_stocks:
        symbol = stock['symbol']
        print(f"\nAnalyzing {symbol}...")
        bars = get_historical_data(symbol)
        if bars is None:
            continue
        df = calculate_technical_indicators(bars)
        options_data = get_options_data(symbol)
        if not options_data:
            print(f"❌ {symbol}: No options data available")
            continue
        analysis = analyze_market_conditions(df, options_data)
        trade_decision = None
        latest = df.iloc[-1]
        # Determine trade direction based on aggregated signals
        if analysis['entry_signals'] == ['strong_bullish']:
            trade_decision = 'CALL'
        elif analysis['entry_signals'] == ['strong_bearish']:
            trade_decision = 'PUT'
        if trade_decision:
            trade_info = {
                'symbol': symbol,
                'price': stock['price'],
                'direction': trade_decision
            }
            filtered_stocks.append(trade_info)
            print(f"✅ {symbol}: {trade_decision} opportunity at ${stock['price']:.2f}")
        else:
            print(f"❌ {symbol}: No clear directional opportunity")
    print(f"\nFound {len(filtered_stocks)} potential trades:")
    for trade in filtered_stocks:
        print(f"{trade['symbol']}: {trade['direction']} at ${trade['price']:.2f}")
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
