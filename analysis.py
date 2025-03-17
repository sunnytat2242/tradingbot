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

# Entry Signal Requirements - UPDATED for more permissive entry
ENTRY_TREND_THRESHOLD = 1    # Reduced from 3 to 1
ENTRY_MOMENTUM_THRESHOLD = 1 # Reduced from 3 to 1
ENTRY_RISK_THRESHOLD = 0     # Reduced from 2 to 0

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
        'volatility_state': '',
        'momentum_score': 0  # Ensure this is initialized
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

    # Additional Momentum Confirmation
    mom_confirm = 0
    if latest['macd'] > latest['macd_signal'] and latest['rsi'] > 55:
        mom_confirm += 1
        print("MACD and RSI both confirm bullish momentum (+1)")
    elif latest['macd'] < latest['macd_signal'] and latest['rsi'] < 45:
        mom_confirm -= 1
        print("MACD and RSI both confirm bearish momentum (-1)")
    momentum_score += mom_confirm
    analysis['momentum_score'] = momentum_score  # Set here


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
    if latest['rsi'] > 80:  # Stricter overbought threshold
        risk_level -= 1
        print("Extreme overbought conditions (-1)")
    elif latest['rsi'] < 20:  # Stricter oversold threshold
        risk_level -= 1
        print("Extreme oversold conditions (-1)")
    if volatility_state == 'low':
        risk_level += 1
    elif volatility_state == 'high':
        risk_level -= 1
    if options_data:
        iv_rank = options_data.get('iv_rank', 50)
        if 20 <= iv_rank <= 75:
            risk_level += 1
            print("Optimal IV range (+1)")
        elif iv_rank > 75:
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
    if abs(trend_score) + abs(momentum_score) >= 6:
        risk_level += 1
        print("Strong trend-momentum synergy (+1)")
    analysis['risk_level'] = risk_level

    # Entry Signals - UPDATED for more permissive entry
    if trend_score >= 1 and momentum_score >= 1:
        analysis['entry_signals'].append('strong_bullish')
        print("Entry Signal: strong_bullish")
    elif trend_score <= -1 and momentum_score <= -1:
        analysis['entry_signals'].append('strong_bearish')
        print("Entry Signal: strong_bearish")
    elif trend_score >= 1 and momentum_score >= 1 and volatility_state != 'high':
        analysis['entry_signals'].append('bullish')
        print("Entry Signal: bullish")
    elif trend_score <= -1 and momentum_score <= -1 and volatility_state != 'high':
        analysis['entry_signals'].append('bearish')
        print("Entry Signal: bearish")

    # Indicator Confluence Signals (More Permissive Criteria)
    if latest['macd'] > latest['macd_signal'] and latest['rsi'] > 50 and trend_score > 0:
        analysis['entry_signals'].append('indicator_confluence_bullish')
        print("Entry Signal: indicator_confluence_bullish")
    elif latest['macd'] < latest['macd_signal'] and latest['rsi'] < 50 and trend_score < 0:
        analysis['entry_signals'].append('indicator_confluence_bearish')
        print("Entry Signal: indicator_confluence_bearish")

    # Exit Signals
    if latest['rsi'] > RSI_EXIT_HIGH or latest['rsi'] < RSI_EXIT_LOW:  # Using defined thresholds (80, 20)
        analysis['exit_signals'].append('rsi_extreme')
        print("Exit Signal: RSI extreme")
    if latest['close'] < latest['sma10'] and prev['close'] > prev['sma10'] and trend_score > 0:
        analysis['exit_signals'].append('trend_reversal')
        print("Exit Signal: Trend reversal (bullish)")
    elif latest['close'] > latest['sma10'] and prev['close'] < prev['sma10'] and trend_score < 0:
        analysis['exit_signals'].append('trend_reversal')
        print("Exit Signal: Trend reversal (bearish)")
    if volatility_state == 'high' and abs(latest['roc']) > (df['roc'].std() * ROC_VOLATILITY_MULTIPLIER):
        analysis['exit_signals'].append('volatility_spike')
        print("Exit Signal: Volatility spike")
    
    # Refined Momentum Loss
    macd_hist_change = (latest['macd_hist'] - prev['macd_hist']) / abs(prev['macd_hist']) if prev['macd_hist'] != 0 else 0
    if (momentum_score > 0 and macd_hist_change < -0.15 and latest['rsi'] < RSI_NEUTRAL_HIGH) or \
       (momentum_score < 0 and macd_hist_change > 0.15 and latest['rsi'] > RSI_NEUTRAL_LOW):
        analysis['exit_signals'].append('momentum_loss')
        print("Exit Signal: Momentum loss")

    # ENHANCED: Price-based exit signals
    if latest['close'] < latest['sma5'] and prev['close'] > prev['sma5'] and trend_score > 0:
        analysis['exit_signals'].append('short_term_trend_break')
        print("Exit Signal: Short-term trend break")

    # ENHANCED: Volume-based exit signals
    if latest['volume_ratio'] < 0.5 and trend_score != 0:
        analysis['exit_signals'].append('volume_decline')
        print("Exit Signal: Volume decline")

    # ENHANCED: Momentum divergence exit
    price_higher = latest['close'] > prev['close']
    momentum_lower = latest['rsi'] < prev['rsi']
    if price_higher and momentum_lower and trend_score > 0:
        analysis['exit_signals'].append('bullish_divergence')
        print("Exit Signal: Bullish divergence")
    elif not price_higher and not momentum_lower and trend_score < 0:
        analysis['exit_signals'].append('bearish_divergence')
        print("Exit Signal: Bearish divergence")

    return analysis
