# Global Parameters for Market Analysis (adjust these as needed)
# Trend Analysis Parameters
SHORT_MA_PERIOD = 5          # for intraday short-term moving average (sma5)
LONG_MA_PERIOD = 10          # for intraday long-term moving average (sma10)
VOLUME_RATIO_THRESHOLD = 1.5 # Threshold for volume confirmation

# Price Action Parameters
# (These depend on how you compute these metrics in your DataFrame)
# No explicit thresholds here; we compare the candle's values to the DataFrame average

# MACD Parameters
MACD_BULLISH_POINTS = 2      # Points added for bullish MACD confirmation
MACD_BEARISH_POINTS = 2      # Points subtracted for bearish MACD confirmation

# RSI Parameters
RSI_NEUTRAL_LOW = 40
RSI_NEUTRAL_HIGH = 60
RSI_EXIT_LOW = 20            # Exit signal if RSI < 20
RSI_EXIT_HIGH = 80           # Exit signal if RSI > 80

# ROC Parameters
# No explicit threshold, just comparing to the mean
# Volatility Analysis Parameters
ATR_LOW_MULTIPLIER = 0.8     # ATR lower bound multiplier
ATR_HIGH_MULTIPLIER = 1.2    # ATR upper bound multiplier

# Risk Assessment Parameters
BOLLINGER_RISK_PENALTY = 1   # risk adjustment for price outside Bollinger Bands
IV_LOW_THRESHOLD = 30        # Low implied volatility threshold
IV_HIGH_THRESHOLD = 70       # High implied volatility threshold
PUT_CALL_RATIO_BULLISH = 0.7 # below this is bullish sentiment
PUT_CALL_RATIO_BEARISH = 1.3 # above this is bearish sentiment

# Entry Signal Requirements
ENTRY_TREND_THRESHOLD = 2
ENTRY_MOMENTUM_THRESHOLD = 2
ENTRY_RISK_THRESHOLD = 1

# Volatility Spike Exit
ROC_VOLATILITY_MULTIPLIER = 2  # multiplier for ROC std dev in volatility spike exit

def analyze_market_conditions_day_trading(df, options_data=None):
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
    if latest['close'] > latest['sma5'] > latest['sma10']:
        trend_score += 2
        print("Strong uptrend confirmed by SMA5 and SMA10 (+2)")
    elif latest['close'] < latest['sma5'] < latest['sma10']:
        trend_score -= 2
        print("Strong downtrend confirmed by SMA5 and SMA10 (-2)")
    if latest['close'] > latest['vwap'] > prev['vwap']:
        trend_score += 1
        print("VWAP breakout confirmed (+1)")
    if latest['volume_ratio'] > 1.5:
        trend_score += 1
        print("Strong volume confirmation (+1)")
    analysis['trend_strength'] = trend_score
    print(f"Trend Score: {trend_score}")

    # Momentum Analysis
    momentum_score = 0
    if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > prev['macd_hist']:
        momentum_score += 2
        print("Strong MACD bullish confirmation (+2)")
    elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < prev['macd_hist']:
        momentum_score -= 2
        print("Strong MACD bearish confirmation (-2)")
    if latest['rsi'] > 60:
        momentum_score += 1
        print("Bullish RSI momentum (+1)")
    elif latest['rsi'] < 40:
        momentum_score -= 1
        print("Bearish RSI momentum (-1)")
    if latest['roc'] > 0 and latest['roc'] > df['roc'].mean():
        momentum_score += 1
        print("Strong positive momentum (+1)")
    elif latest['roc'] < 0 and latest['roc'] < df['roc'].mean():
        momentum_score -= 1
        print("Strong negative momentum (-1)")
    print(f"Momentum Score: {momentum_score}")

    # Volatility Analysis
    current_atr = latest['atr']
    avg_atr = df['atr'].mean()
    if current_atr < avg_atr * 0.8:
        volatility_state = 'low'
        print("Low volatility environment (+1)")
    elif current_atr > avg_atr * 1.3:
        volatility_state = 'high'
        print("High volatility environment (-1)")
    else:
        volatility_state = 'normal'
    analysis['volatility_state'] = volatility_state

    # Risk Assessment
    risk_level = 0
    if latest['rsi'] > 80:
        risk_level -= 1
        print("Extreme overbought conditions (-1)")
    elif latest['rsi'] < 20:
        risk_level -= 1
        print("Extreme oversold conditions (-1)")
    if volatility_state == 'low':
        risk_level += 1
    elif volatility_state == 'high':
        risk_level -= 1
    if options_data:
        iv_rank = options_data.get('iv_rank', 50)
        if 15 <= iv_rank <= 80:
            risk_level += 1
            print("Optimal IV range (+1)")
        elif iv_rank > 80:  # Tightened from 90
            risk_level -= 1
            print("High IV environment (-1)")
        if options_data.get('put_call_ratio', 1) < 0.7:
            risk_level += 1
            print("Bullish options sentiment (+1)")
        elif options_data.get('put_call_ratio', 1) > 1.3:
            risk_level -= 1
            print("Bearish options sentiment (-1)")
    if latest['volume_ratio'] > 1.5:
        risk_level += 1
        print("Volume confirmation boosts risk (+1)")
    if abs(trend_score) + abs(momentum_score) >= 5:
        risk_level += 1
        print("Strong trend-momentum synergy (+1)")
    analysis['risk_level'] = risk_level

    # Entry Signals
    if trend_score >= 2 and momentum_score >= 1 and volatility_state != 'high':
        analysis['entry_signals'].append('strong_bullish')
        print("Entry Signal: strong_bullish")
    elif trend_score <= -2 and momentum_score <= -1 and volatility_state != 'high':
        analysis['entry_signals'].append('strong_bearish')
        print("Entry Signal: strong_bearish")
    elif volatility_state == 'high' and momentum_score >= 3:
        analysis['entry_signals'].append('strong_bullish')
        print("Entry Signal: strong_bullish (high volatility)")
    elif volatility_state == 'high' and momentum_score <= -3:
        analysis['entry_signals'].append('strong_bearish')
        print("Entry Signal: strong_bearish (high volatility)")

    # Exit Signals
    if latest['rsi'] > 80 or latest['rsi'] < 20:
        analysis['exit_signals'].append('rsi_extreme')
        print("Exit Signal: RSI extreme")
    if latest['close'] < latest['sma5'] and prev['close'] > prev['sma5'] and trend_score > 0:
        analysis['exit_signals'].append('trend_reversal')
        print("Exit Signal: Trend reversal (bullish)")
    elif latest['close'] > latest['sma5'] and prev['close'] < prev['sma5'] and trend_score < 0:
        analysis['exit_signals'].append('trend_reversal')
        print("Exit Signal: Trend reversal (bearish)")
    if volatility_state == 'high' and abs(latest['roc']) > (df['roc'].std() * 2):
        analysis['exit_signals'].append('volatility_spike')
        print("Exit Signal: Volatility spike")

    return analysis