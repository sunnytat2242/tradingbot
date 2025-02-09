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
from typing import List, Dict
import requests.exceptions
from config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPHA_VANTAGE_API_KEY
)
from ta.trend import ADXIndicator, IchimokuIndicator
from ta.momentum import StochRSIIndicator
from ta.volume import MFIIndicator
import numpy as np
import yfinance as yf


# Alpaca Configuration

BASE_URL = 'https://paper-api.alpaca.markets'
MARKET_URL = 'https://data.alpaca.markets'


# Initialize client
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL, api_version='v2')

# Trading Parameters
RISK_PER_TRADE = 0.02
MAX_POSITIONS = 10
MIN_STOCK_PRICE = 20.0
MIN_VOLUME = 1000000
MAX_STOCKS_TO_SCAN = 10
API_TIMEOUT = 10
MAX_RETRIES = 3

def get_top_active_stocks(top_n=10):
    """Fetch the top N most active stocks by volume."""
    url = f"{MARKET_URL}/v1beta1/screener/stocks/most-actives"
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_API_SECRET
    }
    params = {
        'top': top_n,
        'by': 'volume'
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'most_actives' in data:
            active_stocks = []
            for stock in data['most_actives']:
                symbol = stock['symbol']
                try:
                    # Fetch the latest trade for the symbol
                    last_trade = api.get_latest_trade(symbol)
                    if last_trade:
                        price = last_trade.price
                    else:
                        price = None
                except Exception as e:
                    price = None
                    print(f"Error fetching price for {symbol}: {e}")
                active_stocks.append({
                    'symbol': symbol,
                    'price': price,
                    'volume': stock['volume'],
                    'trade_count': stock['trade_count']
                })
            return active_stocks
        else:
            print(f"Unexpected data format: {data}")
            return []
    else:
        print(f"Error fetching most active stocks: {response.status_code}")
        return []

def get_top_active_stocks_alpha(top_n=10):
    """Fetch the top N most active stocks by volume using Alpha Vantage API."""

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TOP_GAINERS_LOSERS",
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for HTTP issues
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

def get_active_stocks():
    """Get active stocks using paper trading compatible endpoints"""
    try:
        print("Fetching active stocks...")
        
        # Get list of assets from major exchanges
        assets = api.list_assets(status='active')
        eligible_symbols = [
            asset.symbol for asset in assets 
            if (asset.tradable and 
                asset.shortable and 
                asset.exchange in ['NYSE', 'NASDAQ'] and
                not asset.symbol.startswith(('ETF', 'LETF', 'ETN')) and
                '.' not in asset.symbol)
        ]
        
        print(f"Found {len(eligible_symbols)} eligible symbols")
        
        # Get currently trading assets
        active_stocks = []
        processed = 0
        
        for symbol in eligible_symbols:
            try:
                # Get last trade using basic endpoint
                last_trade = api.get_latest_trade(symbol)
                last_quote = api.get_latest_quote(symbol)
                
                if last_trade and last_quote:
                    price = last_trade.price
                    # Estimate volume using quote size
                    volume = last_quote.bid_size + last_quote.ask_size
                    
                    if price >= MIN_STOCK_PRICE:
                        active_stocks.append({
                            'symbol': symbol,
                            'price': price,
                            'volume': volume,
                            'dollar_volume': price * volume
                        })
                        print(f"Added {symbol}: ${price:.2f}")
                
                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed} of {len(eligible_symbols)} symbols...")
                
                time.sleep(0.2)  # Rate limiting
                
                # Break if we have enough stocks
                if len(active_stocks) >= MAX_STOCKS_TO_SCAN * 2:  # Get extra for filtering
                    break
                
            except Exception as e:
                continue
        
        # Sort by price * volume and get top stocks
        active_stocks.sort(key=lambda x: x['dollar_volume'], reverse=True)
        top_stocks = active_stocks[:MAX_STOCKS_TO_SCAN]
        
        print("\nTop active stocks:")
        for stock in top_stocks:
            print(f"{stock['symbol']}: ${stock['price']:.2f}")
        
        return top_stocks
    
    except Exception as e:
        print(f"Error in get_active_stocks: {e}")
        return []

def get_historical_data(symbol, timeframe='1D', limit=200):
    """Get historical data for paper trading account"""
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            # Use a wider date range to ensure we get enough data
            end = datetime.now(pytz.UTC)
            start = end - timedelta(days=limit * 2)
            
            end_str = end.strftime('%Y-%m-%d')
            start_str = start.strftime('%Y-%m-%d')
            
            print(f"Fetching historical data for {symbol} from {start_str} to {end_str}...")
            
            # Get daily bars
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
    """Calculate enhanced technical indicators for more robust stock selection"""

    #print("# Existing indicators")
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()  # Adding MACD histogram

    #print(" # Enhanced RSI implementation")
    rsi = RSIIndicator(df['close'])
    df['rsi'] = rsi.rsi()
    
    #print("# Bollinger Bands with standard and custom settings")
    bb_20 = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper_20'] = bb_20.bollinger_hband()
    df['bb_lower_20'] = bb_20.bollinger_lband()
    df['bb_mid_20'] = bb_20.bollinger_mavg()
    df['bb_width'] = (df['bb_upper_20'] - df['bb_lower_20']) / df['bb_mid_20']  # Volatility measure
    
    #print("# Enhanced Moving Averages")
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma200'] = df['close'].rolling(window=200).mean()
    
    #print(" # Volume-based indicators")
    df['volume_sma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma20']
    
    #print(" # Price momentum indicators")
    df['roc'] = df['close'].pct_change(periods=12) * 100  # 12-period Rate of Change
    
    #print(" # Volatility indicators")
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    #print(" # Adding Hull Moving Average (HMA) - better at catching trends")
    def hull_moving_average(series, period):
        half_period = int(period/2)
        sqrt_period = int(np.sqrt(period))
        wma1 = series.rolling(window=half_period).mean()
        wma2 = series.rolling(window=period).mean()
        diff = 2 * wma1 - wma2
        return diff.rolling(window=sqrt_period).mean()
    
    df['hma20'] = hull_moving_average(df['close'], 20)
    print("Done with Technical Indicators")
    return df


def analyze_market_conditions(df, options_data=None):
    """Enhanced analysis of market conditions with multiple timeframes and more indicators"""
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
    trend_signals = 0
    if latest['ema9'] > latest['sma20']:
        trend_signals += 1
        print("EMA9 is above SMA20: Bullish trend (+1)")
    else:
        trend_signals -= 1
        print("EMA9 is below SMA20: Bearish trend (-1)")

    if latest['sma20'] > latest['sma50']:
        trend_signals += 1
        print("SMA20 is above SMA50: Bullish trend (+1)")
    else:
        trend_signals -= 1
        print("SMA20 is below SMA50: Bearish trend (-1)")

    if latest['sma50'] > latest['sma200']:
        trend_signals += 1
        print("SMA50 is above SMA200: Long-term uptrend (+1)")
    else:
        trend_signals -= 1
        print("SMA50 is below SMA200: Long-term downtrend (-1)")

    if latest['close'] > latest['hma20']:
        trend_signals += 1
        print("Close price is above HMA20: Bullish momentum (+1)")
    else:
        trend_signals -= 1
        print("Close price is below HMA20: Bearish momentum (-1)")
    
    # Volume Analysis
    volume_confirmed = latest['volume_ratio'] > 1.5
    
    # Momentum Analysis
    momentum_signals = 0
    momentum_signals += 1 if latest['rsi'] > 50 else -1
    momentum_signals += 1 if latest['macd'] > latest['macd_signal'] else -1
    momentum_signals += 1 if latest['roc'] > 0 else -1
    
    # Volatility Analysis
    volatility_state = 'high' if latest['bb_width'] > df['bb_width'].mean() * 1.2 else 'low'
    
    # Risk Analysis
    risk_level = 0
    risk_level += 1 if 30 < latest['rsi'] < 70 else 0  # Non-extreme RSI
    risk_level += 1 if volume_confirmed else 0
    risk_level += 1 if volatility_state == 'low' else 0
    
    # Options Data Integration (if available)
    if options_data and options_data['iv_percentile'] is not None:
        if options_data['iv_percentile'] < 30:  # Low IV environment
            print("iv_percentile is below 30: Lower IV momentum Buy (+1)")
            risk_level += 1
        elif options_data['iv_percentile'] > 70:  # High IV environment
            print("iv_percentile is below 30: Lower IV momentum Risky(-1)")
            risk_level -= 1
    if options_data['put_call_ratio'] is not None:
        if options_data['put_call_ratio'] > 1.5:  # Extremely bearish sentiment
            print("put_call_ratio is above 1.5: trend signals Bearish (-1)")
            trend_signals -= 1  # Market may continue downward
        elif options_data['put_call_ratio'] < 0.4:  # Extremely bullish sentiment
            print("put_call_ratio is below 0.4: trend signals Bullish (+1)")
            trend_signals += 1  # Market may continue upward

    # Entry Conditions
    if trend_signals >= 2 and momentum_signals >= 2 and risk_level >= 3:
        analysis['entry_signals'].append('strong_bullish')
    elif trend_signals <= -2 and momentum_signals <= -2 and risk_level >= 3:
        analysis['entry_signals'].append('strong_bearish')
    
    # Exit Conditions
    if latest['rsi'] > 75 or latest['rsi'] < 25:
        analysis['exit_signals'].append('rsi_extreme')
    if latest['close'] < latest['sma20'] and prev['close'] > prev['sma20']:
        analysis['exit_signals'].append('ma_crossover')
    
    analysis['trend_strength'] = trend_signals
    analysis['risk_level'] = risk_level
    analysis['volatility_state'] = volatility_state
    print("Done with market Conditions")
    return analysis

def get_position_size(analysis, account_size, stock_price):
    """Calculate position size based on risk analysis"""
    base_risk = 0.02  # 2% base risk per trade
    
    # Adjust risk based on analysis
    if analysis['risk_level'] >= 4:
        risk_multiplier = 1.0
    elif analysis['risk_level'] == 3:
        risk_multiplier = 0.75
    else:
        risk_multiplier = 0.5
    
    # Further adjust based on volatility
    if analysis['volatility_state'] == 'high':
        risk_multiplier *= 0.8
    
    # Calculate final position size
    risk_amount = account_size * base_risk * risk_multiplier
    position_size = risk_amount / stock_price
    
    return np.floor(position_size)

def get_options_data(symbol):
    """Fetch option-related data like IV Rank and Put/Call Ratio"""
    try:
        # Fetch the stock data using yfinance
        stock = yf.Ticker(symbol)
        
        # Ensure there are available expiry dates
        if not stock.options:
            print(f"No available option expirations for {symbol}")
            return None

        # Get the first available expiration date
        expiry = stock.options[0]
        
        # Fetch the option chain for the first expiration date
        try:
            options_chain = stock.option_chain(expiry)
            
            # Calculate IV values from the options chains
            call_ivs = options_chain.calls['impliedVolatility'].dropna().values
            put_ivs = options_chain.puts['impliedVolatility'].dropna().values
            
            # Combine all IV values
            iv_values = np.concatenate([call_ivs, put_ivs])
            
            # Calculate IV Percentile only if we have valid IV values
            if len(iv_values) > 0:
                iv_percentile = float(np.percentile(iv_values, 50))
            else:
                iv_percentile = None
            
            # Calculate put/call ratio from volume
            total_call_volume = options_chain.calls['volume'].sum()
            total_put_volume = options_chain.puts['volume'].sum()
            
            if total_call_volume > 0:
                put_call_ratio = float(total_put_volume / total_call_volume)
            else:
                put_call_ratio = None

            return {
                'iv_percentile': iv_percentile,
                'put_call_ratio': put_call_ratio
            }
            
        except (AttributeError, TypeError) as e:
            print(f"Error processing options chain for {symbol}: {e}")
            return None

    except Exception as e:
        print(f"Error fetching options data for {symbol}: {e}")
        return None


def get_stock_sentiment_alpha(ticker):
    """Fetch news sentiment for a specific stock ticker using Alpha Vantage."""

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "feed" not in data:
            print(f"Unexpected response: {data}")
            return []

        sentiment_data = []

        for article in data["feed"]:
            sentiment_data.append({
                "headline": article["title"],
                "sentiment": article["overall_sentiment_label"],  # bullish, bearish, neutral
                "relevance": article["relevance_score"],
                "source": article["source"],
                "published": article["time_published"]
            })

        return sentiment_data

    except requests.RequestException as e:
        print(f"Error fetching sentiment data: {e}")
        return []
def get_filtered_stocks(top_n):
    """Filter stocks for directional options trading (calls/puts only)"""
    active_stocks = get_top_active_stocks_alpha(top_n)
    filtered_stocks = []
    
    print("\nAnalyzing stocks for directional options trades...")
    
    for stock in active_stocks:
        symbol = stock['symbol']
        print(f"\nAnalyzing {symbol}...")
        
        # Fetch historical data
        bars = get_historical_data(symbol)
        if bars is None:
            continue
            
        # Calculate enhanced technical indicators
        df = calculate_technical_indicators(bars)
        
        # Get options data
        options_data = get_options_data(symbol)
        if not options_data:
            print(f"❌ {symbol}: No options data available")
            continue
        #market_sentiment = get_stock_sentiment_alpha(symbol)
        # Get market analysis
        analysis = analyze_market_conditions(df, options_data)
        
        # Determine if we should trade and in which direction
        trade_decision = None
        latest = df.iloc[-1]
        
        # BULLISH SIGNALS (CALLS)
        if (analysis['trend_strength'] >= 2 and  # Strong upward trend
                30 < latest['rsi'] < 70 and  # RSI not overbought
            latest['macd'] > latest['macd_signal']):  # Good volume
            trade_decision = 'CALL'
            
        # BEARISH SIGNALS (PUTS)
        elif (analysis['trend_strength'] <= -2 and  # Strong downward trend
              60 > latest['rsi'] > 35 and  # RSI not oversold
              latest['macd'] < latest['macd_signal']):  # Good volume
            trade_decision = 'PUT'
            
        # Add stock if it meets our criteria
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
    
    # Sort by trend strength (using the absolute value for both calls and puts)
    filtered_stocks.sort(key=lambda x: abs(x.get('trend_strength', 0)), reverse=True)
    
    print(f"\nFound {len(filtered_stocks)} potential trades:")
    for trade in filtered_stocks:
        print(f"{trade['symbol']}: {trade['direction']} at ${trade['price']:.2f}")
        
    return filtered_stocks


def options_trading_bot():
    """Main options trading loop"""
    while True:
        try:
            print("\n=== Scanning for Options Trading Opportunities ===")

            filtered_stocks = get_filtered_stocks()
            if not filtered_stocks:
                print("No stocks meet criteria. Retrying in 5 minutes...")
                time.sleep(300)
                continue

            for stock in filtered_stocks:
                symbol = stock['symbol']
                print(f"\nChecking options for {symbol}")

                options_data = get_options_data(symbol)
                if options_data is None:
                    continue

                # Trading Rules
                if options_data['iv_percentile'] > 50 and options_data['put_call_ratio'] > 1.0:
                    print(f"⚠️ High IV & Bearish Sentiment: Selling premium on {symbol}")
                elif options_data['iv_percentile'] < 20 and options_data['put_call_ratio'] < 0.7:
                    print(f"🚀 Low IV & Bullish Sentiment: Buying calls on {symbol}")

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