import asyncio
import aiohttp
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

# Trading Parameters - ENHANCED for better risk management
RISK_PER_TRADE = 0.01  # Reduced from 0.02 to 0.01
MIN_STOCK_PRICE = 20.0
MAX_STOCKS_TO_SCAN = 100
API_TIMEOUT = 10
MAX_RETRIES = 10

# Global Parameters - UPDATED for more permissive entry
ENTRY_TREND_THRESHOLD = 1    # Reduced from 2 to 1
ENTRY_MOMENTUM_THRESHOLD = 1 # Reduced from 2 to 1
ENTRY_RISK_THRESHOLD = 0     # Reduced from 1 to 0

# ENHANCED: Time-based filters
BEST_TRADING_DAYS = [1, 2, 4]  # Monday, Tuesday, Thursday (0=Monday)

options_stocks = ['TSLA', 'PLTR', 'NVDA', 'AAPL', 'AMZN', 'ASML', 'AVGO', 'COIN', 'GOOG', 'META', 'MSTR', 'NFLX',
                  'ORCL', 'QDTE', 'SPY', 'SMCI', 'UNH', 'QQQ', 'LRCX', 'JPM', 'SNOW', 'FI', 'BA', 'MELI', 'INTC', 'AMD', 'MARA', 'MSFT']

# Cache for historical data to reduce API calls
historical_data_cache = {}
options_data_cache = {}

# ENHANCED: Sector mapping for risk management
sector_mapping = {
    'TSLA': 'Consumer Discretionary',
    'PLTR': 'Technology',
    'NVDA': 'Technology',
    'AAPL': 'Technology',
    'AMZN': 'Consumer Discretionary',
    'ASML': 'Technology',
    'AVGO': 'Technology',
    'COIN': 'Financials',
    'GOOG': 'Communication Services',
    'META': 'Communication Services',
    'MSTR': 'Technology',
    'NFLX': 'Communication Services',
    'ORCL': 'Technology',
    'QDTE': 'Technology',
    'SPY': 'ETF',
    'SMCI': 'Technology',
    'UNH': 'Healthcare',
    'QQQ': 'ETF',
    'LRCX': 'Technology',
    'JPM': 'Financials',
    'SNOW': 'Technology',
    'FI': 'Technology',
    'BA': 'Industrials',
    'MELI': 'Consumer Discretionary',
    'INTC': 'Technology',
    'AMD': 'Technology',
    'MARA': 'Technology',
    'MSFT': 'Technology'
}

async def get_stock_data_async(symbols):
    """Async version of get_stock_data to fetch quotes in parallel"""
    stock_data = []
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_API_SECRET
    }
    
    async def fetch_single_stock(session, symbol):
        endpoint = f"{MARKET_URL}/v2/stocks/{symbol}/quotes/latest"
        try:
            async with session.get(endpoint, headers=headers, timeout=API_TIMEOUT) as response:
                if response.status == 200:
                    data = await response.json()
                    try:
                        price = data["quote"]["ap"]  # Ask price
                        volume = data["quote"]["as"]
                        return {"symbol": symbol, "price": price, "volume": volume}
                    except KeyError:
                        return {"symbol": symbol, "error": "Data unavailable"}
                else:
                    return {"symbol": symbol, "error": f"Status code: {response.status}"}
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}
    
    async with aiohttp.ClientSession()  as session:
        semaphore = asyncio.Semaphore(20)  # Increased to 20 for Alpaca
        async def bounded_fetch(symbol):
            async with semaphore:
                return await fetch_single_stock(session, symbol)
        tasks = [bounded_fetch(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
    return [result for result in results if result]

async def get_historical_data_async(symbol, timeframe='5Min', limit=50):
    """Async wrapper for get_historical_data"""
    return await asyncio.to_thread(get_historical_data, symbol, timeframe='5Min', limit=50)

def get_historical_data(symbol, timeframe='5Min', limit=50):
    """Async version of get_historical_data with caching and threading"""
    cache_key = f"{symbol}_{timeframe}"
    current_time = datetime.now(pytz.UTC)
    if cache_key in historical_data_cache:
        cache_time, cached_data = historical_data_cache[cache_key]
        if (current_time - cache_time).total_seconds() < 300:  # 5 minutes
            return cached_data
    
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            end = current_time
            start = end - timedelta(days=5)
            while start.weekday() >= 5:  # Avoid weekends
                start -= timedelta(days=1)
            start_str, end_str = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
            print(f"Fetching historical data for {symbol} from {start_str} to {end_str}...")
            bars = api.get_bars(symbol, timeframe, start=start_str, end=end_str, adjustment='raw', feed='iex')
            bars_df = bars.df
            if len(bars_df) < limit:
                print(f"Warning: Only got {len(bars_df)} bars for {symbol}")
                return None
            historical_data_cache[cache_key] = (current_time, bars_df)
            return bars_df
        except Exception as e:
            print(e)
            retry_count += 1
            time.sleep(1)  # Add delay between retries
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
    sma_windows = [5, 10, 20, 50, 200]
    for window in sma_windows:
        df[f'sma{window}'] = df['close'].rolling(window=window).mean()
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=1).volume_weighted_average_price()
    macd_indicator = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['roc'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
    df.dropna(subset=['sma5', 'sma10', 'macd', 'rsi', 'atr', 'vwap'], inplace=True)
    return df

async def calculate_technical_indicators_async(df):
    """Async wrapper for calculate_technical_indicators"""
    return await asyncio.to_thread(calculate_technical_indicators, df)

def get_position_size(analysis, account_size, stock_price):
    """Calculate position size with enhanced risk management."""
    # ENHANCED: More conservative base risk
    base_risk = RISK_PER_TRADE * 0.7  # Reduce base risk by 30%
    
    # ENHANCED: Stricter criteria for position sizing multipliers
    if analysis['risk_level'] >= 2 and analysis['trend_strength'] >= 3:  # Increased from 2 to 3
        risk_multiplier = 1.0
    elif analysis['risk_level'] >= 1 and analysis['trend_strength'] >= 2:  # Increased from 1 to 2
        risk_multiplier = 0.75
    else:
        risk_multiplier = 0.5
        
    # ENHANCED: More conservative volatility adjustments
    if analysis['volatility_state'] == 'high':
        risk_multiplier *= 0.5  # Reduced from 0.6 to 0.5
    elif analysis['volatility_state'] == 'normal':
        risk_multiplier *= 0.7  # Reduced from 0.8 to 0.7
        
    risk_amount = account_size * base_risk * risk_multiplier
    position_size = risk_amount / stock_price
    min_position = 100
    
    if position_size * stock_price < min_position:
        return 0
        
    return np.floor(position_size)

@sleep_and_retry
@limits(calls=CALLS_PER_HOUR, period=PERIOD_IN_SECONDS)
async def get_options_data_async(symbol, percentile=50):
    """Async version of get_options_data with caching and threading"""
    current_time = datetime.now(pytz.UTC)
    if symbol in options_data_cache:
        cache_time, cached_data = options_data_cache[symbol]
        if (current_time - cache_time).total_seconds() < 900:  # 15 minutes
            return cached_data
    
    async with asyncio.Semaphore(CALLS_PER_MINUTE):
        await asyncio.sleep(60 / CALLS_PER_MINUTE)
        try:
            stock = await asyncio.to_thread(yf.Ticker, symbol)
            if not stock.options:
                print(f"No available option expirations for {symbol}")
                return None
            expiry = stock.options[0]
            options_chain = await asyncio.to_thread(stock.option_chain, expiry)
            underlying_price = stock.info.get("regularMarketPrice")
            if underlying_price is None:
                quote = await asyncio.to_thread(stock.history, period="1d")
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
            iv_percentile = float(np.percentile(current_ivs, percentile) / np.max(current_ivs) * 100 if len(current_ivs) > 0 else 50.0)
            iv_hist = []
            hist = await asyncio.to_thread(stock.history, period="30d")
            for exp in stock.options[:min(3, len(stock.options))]:
                chain = await asyncio.to_thread(stock.option_chain, exp)
                iv_hist.extend(chain.calls['impliedVolatility'].dropna().values)
                iv_hist.extend(chain.puts['impliedVolatility'].dropna().values)
            iv_rank = float(np.percentile(current_ivs, percentile) / np.percentile(iv_hist, 100) * 100 if iv_hist else 50.0)
            iv_rank = min(max(iv_rank, 0), 100)
            total_call_volume = options_chain.calls['volume'].sum()
            total_put_volume = options_chain.puts['volume'].sum()
            put_call_ratio = float(total_put_volume / total_call_volume) if total_call_volume > 0 else 1.0
            avg_call_volume = options_chain.calls['volume'].mean()
            avg_put_volume = options_chain.puts['volume'].mean()
            avg_call_oi = options_chain.calls['openInterest'].mean()
            avg_put_oi = options_chain.puts['openInterest'].mean()
            result = {
                'iv_percentile': iv_percentile,
                'iv_rank': iv_rank,
                'put_call_ratio': put_call_ratio,
                'nearest_strike': nearest_strike,
                'underlying_price': underlying_price,
                'avg_call_volume': avg_call_volume,
                'avg_put_volume': avg_put_volume,
                'avg_call_oi': avg_call_oi,
                'avg_put_oi': avg_put_oi
            }
            options_data_cache[symbol] = (current_time, result)
            print(f"{symbol} Options: IV Percentile={iv_percentile:.1f}, IV Rank={iv_rank:.1f}, Put/Call={put_call_ratio:.2f}")
            return result
        except Exception as e:
            print(f"Error fetching options data for {symbol}: {e}")
            max_retries = 3
            retry_delay = 60
            for retry in range(max_retries):
                await asyncio.sleep(retry_delay)
                try:
                    return await get_options_data_async(symbol, percentile)
                except Exception as retry_e:
                    print(f"Retry {retry + 1} failed: {retry_e}")
                    retry_delay *= 2
            return None

async def analyze_single_stock(stock, market_trend):
    try:
        print(f"\nAnalyzing {stock}...")
        bars = await get_historical_data_async(stock)
        if bars is None:
            print(f"❌ {stock}: No historical data")
            return None
        df = await calculate_technical_indicators_async(bars)
        if df is None:
            print(f"❌ {stock}: No technical indicators")
            return None
        options_data = await get_options_data_async(stock)
        if not options_data:
            print(f"❌ {stock}: No options data available")
            return None

        analysis = analyze_market_conditions_day_trading(df, options_data)
        print(f"Analysis returned: {analysis}")  # Debug full analysis

        # UPDATED: More permissive trade decision logic
        trade_decision = None
        
        # More permissive criteria for bullish trades
        if market_trend == 'bullish':
            if (('strong_bullish' in analysis['entry_signals'] or 'indicator_confluence_bullish' in analysis['entry_signals']) and 
                analysis['risk_level'] >= ENTRY_RISK_THRESHOLD and 
                (analysis['trend_strength'] >= ENTRY_TREND_THRESHOLD or analysis['momentum_score'] >= ENTRY_MOMENTUM_THRESHOLD)):
                trade_decision = 'CALL'
        # More permissive criteria for bearish trades
        elif market_trend == 'bearish':
            if (('strong_bearish' in analysis['entry_signals'] or 'indicator_confluence_bearish' in analysis['entry_signals']) and 
                analysis['risk_level'] >= ENTRY_RISK_THRESHOLD and 
                (analysis['trend_strength'] <= -ENTRY_TREND_THRESHOLD or analysis['momentum_score'] <= -ENTRY_MOMENTUM_THRESHOLD)):
                trade_decision = 'PUT'
        # Allow counter-trend trades with very strong signals
        else:
            if ('strong_bearish' in analysis['entry_signals'] and 'indicator_confluence_bearish' in analysis['entry_signals'] and
                analysis['trend_strength'] <= -3 and analysis['momentum_score'] <= -3 and analysis['risk_level'] >= 2):
                trade_decision = 'PUT'
            elif ('strong_bullish' in analysis['entry_signals'] and 'indicator_confluence_bullish' in analysis['entry_signals'] and
                  analysis['trend_strength'] >= 3 and analysis['momentum_score'] >= 3 and analysis['risk_level'] >= 2):
                trade_decision = 'CALL'

        # Always return a stock dict, whether tradeable or not
        stock_info = {
            'symbol': stock,
            'price': options_data['underlying_price'],
            'direction': trade_decision if trade_decision else ('PUT' if analysis['trend_strength'] < 0 else 'CALL'),
            'nearest_strike': options_data['nearest_strike'],
            'trend_strength': analysis['trend_strength'],
            'momentum_score': analysis['momentum_score'],
            'risk_level': analysis['risk_level'],  # Include always
            'entry_signals': analysis['entry_signals'],  # Include always
            'market_trend': market_trend,
            'options_data': options_data,
            'sector': sector_mapping.get(stock, 'Other')
        }

        if trade_decision:
            message = f"✅ {stock}: {trade_decision} opportunity at ${options_data['underlying_price']:.2f}, strike ${options_data['nearest_strike']:.2f}"
            await send_telegram_alert(message)
            print(f"Trade info prepared: {stock_info}")
        else:
            if analysis['exit_signals']:
                print(f"❌ {stock}: No clear directional opportunity - Exit signals: {', '.join(analysis['exit_signals'])}")
            elif analysis['risk_level'] < ENTRY_RISK_THRESHOLD:
                print(f"❌ {stock}: No clear directional opportunity - Risk level {analysis['risk_level']} below threshold {ENTRY_RISK_THRESHOLD}")
            elif analysis['trend_strength'] < ENTRY_TREND_THRESHOLD and analysis['trend_strength'] > -ENTRY_TREND_THRESHOLD:
                print(f"❌ {stock}: No clear directional opportunity - Trend strength {analysis['trend_strength']} below threshold {ENTRY_TREND_THRESHOLD}")
            elif abs(analysis['momentum_score']) < ENTRY_MOMENTUM_THRESHOLD:
                print(f"❌ {stock}: No clear directional opportunity - Momentum score {analysis['momentum_score']} below threshold {ENTRY_MOMENTUM_THRESHOLD}")
            else:
                print(f"❌ {stock}: No clear directional opportunity - Market trend mismatch (Stock: {trade_decision}, Market: {market_trend})")

        return stock_info  # Return full info always

    except Exception as e:
        print(f"Error analyzing {stock}: {e}")
        return None

async def get_filtered_stocks_async():
    print("Analyzing stocks for directional options trades...")
    
    # ENHANCED: Get SPY data for market trend with better error handling
    spy_bars = await get_historical_data_async('SPY')
    if spy_bars is None:
        print("Unable to get SPY data for market trend analysis. Using neutral trend.")
        market_trend = 'neutral'
    else:
        spy_df = await calculate_technical_indicators_async(spy_bars)
        if spy_df is None:
            print("Unable to calculate SPY indicators. Using neutral trend.")
            market_trend = 'neutral'
        else:
            # Enhanced market trend analysis
            spy_analysis = analyze_market_conditions_day_trading(spy_df)
            
            # More nuanced market trend determination
            if spy_analysis['trend_strength'] >= 2 and spy_analysis['momentum_score'] >= 1:
                market_trend = 'bullish'
            elif spy_analysis['trend_strength'] <= -2 and spy_analysis['momentum_score'] <= -1:
                market_trend = 'bearish'
            else:
                # Check additional market indicators
                if spy_df.iloc[-1]['close'] > spy_df.iloc[-1]['vwap']:
                    if spy_df.iloc[-1]['rsi'] > 50:
                        market_trend = 'bullish'
                    else:
                        market_trend = 'neutral'
                else:
                    if spy_df.iloc[-1]['rsi'] < 50:
                        market_trend = 'bearish'
                    else:
                        market_trend = 'neutral'
    
    print(f"Market trend determined as: {market_trend}")

    stocks = ['TSLA', 'PLTR', 'NVDA', 'AAPL', 'AMZN', 'ASML', 'AVGO', 'COIN', 'GOOG', 'META', 
              'MSTR', 'NFLX', 'ORCL', 'QDTE', 'SPY', 'SMCI', 'UNH', 'QQQ', 'LRCX', 'JPM', 
              'SNOW', 'FI', 'BA', 'MELI', 'INTC', 'AMD', 'MARA', 'MSFT']
    tasks = [analyze_single_stock(stock, market_trend) for stock in stocks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    filtered_stocks = []
    for r in results:
        if isinstance(r, Exception):
            print(f"Skipping result due to exception: {r}")
        elif r is None:
            continue
        elif not isinstance(r, dict) or 'momentum_score' not in r:
            print(f"Invalid trade info format: {r}")
        else:
            # Debug: Inspect raw result from analyze_single_stock
            print(f"DEBUG: Raw analysis for {r['symbol']}: {r}")
            
            # Build stock dict with all required keys
            stock = {
                'symbol': r['symbol'],
                'price': r.get('price'),  # Should come from analyze_single_stock
                'direction': r.get('direction', 'PUT' if r['trend_strength'] < 0 else 'CALL'),
                'trend_strength': r['trend_strength'],
                'momentum_score': r['momentum_score'],
                'risk_level': r['risk_level'],  # Direct access, no default
                'entry_signals': r['entry_signals'],  # Direct access, no default
                'market_trend': market_trend,
                'options_data': r.get('options_data', {}),
                'sector': r.get('sector', 'Unknown')
            }
            print(f"Valid trade info for {r['symbol']}: {stock}")
            filtered_stocks.append(stock)

    if not filtered_stocks:
        print("No valid trade candidates after filtering.")
        return []

    diversified_stocks = sorted(filtered_stocks, 
                                key=lambda x: (abs(x['trend_strength']) + abs(x['momentum_score'])) * 
                                              (1.5 if x['market_trend'] == x['direction'].lower() else 1.0),
                                reverse=True)[:10]
    return diversified_stocks



async def options_trading_bot_async():
    """Async version of options_trading_bot with improved filtering"""
    while True:
        try:
            # ENHANCED: Time-based filters
            current_time = datetime.now(pytz.UTC)
            current_hour = current_time.hour
            current_weekday = current_time.weekday()
            
                
            print("\n=== Scanning for Options Trading Opportunities ===")
            filtered_stocks = await get_filtered_stocks_async()
            if not filtered_stocks:
                print("No stocks meet criteria. Retrying in 5 minutes...")
                await asyncio.sleep(300)
                continue
            tasks = [check_options_for_stock(stock['symbol'], stock) for stock in filtered_stocks]
            await asyncio.gather(*tasks)
            print("\nScan complete. Waiting for next scan...")
            await asyncio.sleep(300)
        except KeyboardInterrupt:
            print("\nStopping options bot...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            await asyncio.sleep(60)

async def check_options_for_stock(symbol, stock_data):
    """Check options data for a single stock using cached data"""
    try:
        print(f"\nChecking options for {symbol}...")
        options_data = stock_data.get('options_data')  # Reuse data from analyze_single_stock
        if options_data is None:
            options_data = await get_options_data_async(symbol)  # Fallback to fetch if missing
            if options_data is None:
                return
                
        # ENHANCED: More conservative IV thresholds
        if (options_data.get('iv_percentile') is not None and 
            options_data['iv_percentile'] > 65 and  # Lowered from 70
            options_data.get('put_call_ratio') is not None and 
            options_data['put_call_ratio'] > 1.2):
            message = f"⚠️ {symbol}: High IV ({options_data['iv_percentile']:.1f}%) & Bearish Sentiment: Consider selling premium"
            print(message)
            await send_telegram_alert(message)
        elif (options_data.get('iv_percentile') is not None and 
              options_data['iv_percentile'] < 30 and
              options_data.get('put_call_ratio') is not None and 
              options_data['put_call_ratio'] < 0.7 and
              stock_data.get('market_trend') == 'bullish'):  # Require stronger trend alignment
            message = f"🚀 {symbol}: Low IV ({options_data['iv_percentile']:.1f}%) & Bullish Sentiment: Consider buying calls"
            print(message)
            await send_telegram_alert(message)
    except Exception as e:
        print(f"Error checking options for {symbol}: {e}")

def get_filtered_stocks(top_n=10):
    """Synchronous wrapper for get_filtered_stocks_async"""
    return asyncio.run(get_filtered_stocks_async())

def options_trading_bot():
    """Synchronous wrapper for options_trading_bot_async"""
    return asyncio.run(options_trading_bot_async())

if __name__ == '__main__':
    options_trading_bot()
