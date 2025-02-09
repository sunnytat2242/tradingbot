# Add new imports at the top
from ta.volatility import AverageTrueRange
from optionsbotEnhancedVersion1 import(
    get_historical_data,
    calculate_technical_indicators,
    analyze_market_conditions,
    get_filtered_stocks
)
def detect_candlestick_patterns(df):
    """Detect reversal candlestick patterns in the latest data."""
    patterns = []
    if len(df) < 3:
        return patterns

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    second_prev = df.iloc[-3]

    # Bullish Engulfing Pattern
    if (latest['open'] < prev['close'] < prev['open'] < latest['close'] and
        latest['close'] > latest['open']):
        patterns.append('bullish_engulfing')

    # Bearish Engulfing Pattern
    if (latest['open'] > prev['close'] > prev['open'] > latest['close'] and
        latest['close'] < latest['open']):
        patterns.append('bearish_engulfing')

    # Hammer Pattern
    if ((latest['close'] > latest['open']) and
        (latest['low'] < latest['open'] - (latest['close'] - latest['open'])) and
        (latest['high'] - latest['close']) < (latest['close'] - latest['open'])):
        patterns.append('hammer')

    # Shooting Star
    if ((latest['close'] < latest['open']) and
        (latest['high'] > latest['open'] + (latest['open'] - latest['close'])) and
        (latest['close'] - latest['low']) < (latest['open'] - latest['close'])):
        patterns.append('shooting_star')

    return patterns

def check_divergence(df, window=14):
    """Check for RSI and MACD divergence."""
    divergence = {
        'rsi_bullish': False,
        'rsi_bearish': False,
        'macd_bullish': False,
        'macd_bearish': False
    }

    # RSI Divergence
    rsi = df['rsi'].values
    prices = df['close'].values

    for i in range(len(df)-window, len(df)-2):
        # Bullish divergence (price lower low, RSI higher low)
        if prices[i] < prices[i - 1] and rsi[i] > rsi[i - 1]:
            divergence['rsi_bullish'] = True
        # Bearish divergence (price higher high, RSI lower high)
        if prices[i] > prices[i - 1] and rsi[i] < rsi[i - 1]:
            divergence['rsi_bearish'] = True

    # MACD Divergence
    macd_hist = df['macd_hist'].values
    for i in range(len(df)-window, len(df)-2):
        # Bullish divergence (price lower low, MACD higher low)
        if prices[i] < prices[i - 1] and macd_hist[i] > macd_hist[i - 1]:
            divergence['macd_bullish'] = True
        # Bearish divergence (price higher high, MACD lower high)
        if prices[i] > prices[i - 1] and macd_hist[i] < macd_hist[i - 1]:
            divergence['macd_bearish'] = True

    return divergence

def calculate_technical_indicators_enhanced(df):
    """Add Average True Range for volatility measurement."""
    df = calculate_technical_indicators(df)  # Original calculations

    # Add ATR
    atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr_indicator.average_true_range()

    return df

def analyze_market_conditions_enhanced(df, options_data=None):
    """Enhanced analysis with reversal detection."""
    # Original analysis
    analysis = analyze_market_conditions(df, options_data)  # Existing analysis

    # Add reversal detection
    candlestick_patterns = detect_candlestick_patterns(df)
    divergence = check_divergence(df)

    # Add new signals
    reversal_signals = []

    # Bullish reversal conditions
    if ('bullish_engulfing' in candlestick_patterns or
        'hammer' in candlestick_patterns or
        divergence['rsi_bullish'] or
        divergence['macd_bullish']):
        if df['close'].iloc[-1] > df['sma20'].iloc[-1]:  # Confirm with SMA
            reversal_signals.append('bullish_reversal')

    # Bearish reversal conditions
    if ('bearish_engulfing' in candlestick_patterns or
        'shooting_star' in candlestick_patterns or
        divergence['rsi_bearish'] or
        divergence['macd_bearish']):
        if df['close'].iloc[-1] < df['sma20'].iloc[-1]:  # Confirm with SMA
            reversal_signals.append('bearish_reversal')

    # Update entry signals
    for signal in reversal_signals:
        if signal not in analysis['entry_signals']:
            analysis['entry_signals'].append(signal)

    # Add volatility check using ATR
    atr_current = df['atr'].iloc[-1]
    atr_avg = df['atr'].mean()
    analysis['volatility_state'] = 'high' if atr_current > atr_avg * 1.2 else 'low'

    return analysis

def get_filtered_stocks_enhanced(top_n=10):
    """Updated filtering with reversal signals."""
    filtered_stocks = get_filtered_stocks(top_n)  # Original filtering

    for stock in filtered_stocks:
        symbol = stock['symbol']
        bars = get_historical_data(symbol)
        df = calculate_technical_indicators_enhanced(bars)
        analysis = analyze_market_conditions_enhanced(df)

        # Enhance trade decision with reversal signals
        if 'bullish_reversal' in analysis['entry_signals']:
            stock['direction'] = 'CALL'
        elif 'bearish_reversal' in analysis['entry_signals']:
            stock['direction'] = 'PUT'

    return filtered_stocks