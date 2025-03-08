import asyncio
import re
import alpaca_trade_api as tradeapi
from typing import Dict, List, Tuple
import time
from datetime import datetime, timedelta
import pytz
import math
import schedule
from alerts import send_telegram_alert
from config import ALPACA_API_KEY, ALPACA_API_SECRET
from optionsbotEnhancedVersion1 import get_filtered_stocks
import requests
from scipy.stats import norm

# Trading Parameters (unchanged)
POSITION_SIZE = 0.02
MAX_POSITIONS = 10
MIN_STOCK_PRICE = 25.0
STOP_LOSS = 15.0
INITIAL_PROFIT_TARGET = 20.0
TRAILING_BUFFER_PERCENT = 5.0
PROFIT_TARGET = 25.0

# API Configuration (unchanged)
BASE_URL = 'https://paper-api.alpaca.markets'
HEADERS = {
    'APCA-API-KEY-ID': ALPACA_API_KEY,
    'APCA-API-SECRET-KEY': ALPACA_API_SECRET,
    'Content-Type': 'application/json'
}

class OptionsTradeExecutor:
    def __init__(self):
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL, api_version='v2')
        self.traded_today = set()
        self.last_trade_date = datetime.now(pytz.UTC).date()

    def reset_daily_trades(self):
        """Reset daily traded symbols if a new day starts."""
        current_date = datetime.now(pytz.UTC).date()
        if current_date != self.last_trade_date:
            print("New day detected. Resetting traded symbols list.")
            self.traded_today.clear()
            self.last_trade_date = current_date

    def get_option_contracts(self, symbol: str) -> List[Dict]:
        """Fetch all option contracts for a given symbol using Alpaca API."""
        try:
            url = f"{BASE_URL}/v2/options/contracts"
            params = {
                'underlying_symbols': symbol,
                'status': 'active',
                'limit': 100
            }
            response = requests.get(url, headers=HEADERS, params=params)
            if response.status_code == 200:
                contracts = response.json().get('option_contracts', [])
                print(f"Fetched {len(contracts)} option contracts for {symbol}")
                return contracts
            else:
                print(f"Error fetching option contracts for {symbol}: {response.text}")
                return []
        except Exception as e:
            print(f"Exception fetching option contracts for {symbol}: {e}")
            return []

    def calculate_delta(self, stock_price, strike_price, days_to_expiry, is_call):
        """Calculate the delta of an option using Black-Scholes."""
        T = max(days_to_expiry / 365.0, 0.001)  # Convert days to years, ensure T > 0
        S, K, sigma, r = stock_price, strike_price, 0.3, 0.01  # Volatility and risk-free rate
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return norm.cdf(d1) if is_call else norm.cdf(d1) - 1

    def select_nearest_expiry_and_strike(self, symbol: str, current_price: float, is_bullish: bool) -> Tuple[str, float, datetime]:
        """Select the nearest expiration and strike with delta closest to target from Alpaca API data."""
        contracts = self.get_option_contracts(symbol)
        if not contracts:
            print(f"No option contracts found for {symbol}")
            return None, None, None

        # Filter contracts by type (call or put)
        option_type = 'call' if is_bullish else 'put'
        relevant_contracts = [c for c in contracts if c['type'] == option_type]

        if not relevant_contracts:
            print(f"No {option_type} contracts found for {symbol}")
            return None, None, None

        # Convert expiration dates to datetime and find nearest future expiry
        now = datetime.now(pytz.UTC)
        expiry_dates = sorted(
            set(datetime.strptime(c['expiration_date'], '%Y-%m-%d').replace(tzinfo=pytz.UTC) for c in relevant_contracts),
            key=lambda x: (x - now).days if (x - now).days >= 0 else float('inf')
        )
        if not expiry_dates:
            print(f"No future expirations found for {symbol}")
            return None, None, None

        nearest_expiry = expiry_dates[0]
        print(f"Nearest expiry for {symbol}: {nearest_expiry.strftime('%Y-%m-%d')}")

        # Calculate days to expiry once, before the loop
        days_to_expiry = (nearest_expiry - now).days
        if days_to_expiry < 0:
            print(f"Calculated days_to_expiry {days_to_expiry} is negative for {symbol}")
            return None, None, None
        print(f"Days to expiry for {symbol}: {days_to_expiry}")

        # Filter contracts for this expiry
        expiry_contracts = [c for c in relevant_contracts if c['expiration_date'] == nearest_expiry.strftime('%Y-%m-%d')]
        if not expiry_contracts:
            print(f"No contracts found for expiry {nearest_expiry.strftime('%Y-%m-%d')}")
            return None, None, None

        # Calculate delta for each strike
        strike_deltas = []
        for c in expiry_contracts:
            strike = float(c['strike_price'])
            delta = self.calculate_delta(current_price, strike, days_to_expiry, is_bullish)
            strike_deltas.append((strike, delta, c['symbol']))

        # Select strike with delta closest to target (0.3 for calls, -0.3 for puts)
        target = 0.3 if is_bullish else -0.3
        closest_strike_data = min(strike_deltas, key=lambda x: abs(x[1] - target))
        print(f"Selected strike ${closest_strike_data[0]:.2f} for {symbol} (symbol: {closest_strike_data[2]}, delta: {closest_strike_data[1]:.3f})")

        return closest_strike_data[2], closest_strike_data[0], nearest_expiry

    def calculate_option_price(self, stock_price: float, strike_price: float,
                               days_to_expiry: int, is_call: bool,
                               volatility: float = 0.3, risk_free_rate: float = 0.01) -> float:
        """Calculate theoretical option price using Black-Scholes (unchanged)."""
        T = max(days_to_expiry / 365.0, 0.001)
        if T <= 0:
            return max(0, (stock_price - strike_price) if is_call else (strike_price - stock_price))
        S, K, sigma, r = stock_price, strike_price, volatility, risk_free_rate
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if is_call:
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return round(max(price, 0.01), 2)

    def select_option_contract(self, symbol: str, current_price: float, is_bullish: bool, strike: float) -> Tuple[str, float, float]:
        """Select an option contract using Alpaca API data."""
        try:
            # Fetch nearest expiry and valid strike
            option_symbol, strike_price, expiry_date = self.select_nearest_expiry_and_strike(symbol, current_price, is_bullish)
            if not option_symbol:
                print(f"Could not find suitable option contract for {symbol}")
                return None, None, None

            # Calculate days to expiry
            days_to_expiry = (expiry_date - datetime.now(pytz.UTC)).days
            if days_to_expiry < 0:
                print(f"Selected expiry {expiry_date.strftime('%Y-%m-%d')} is in the past")
                return None, None, None

            # Calculate option price using your Black-Scholes model
            option_price = self.calculate_option_price(current_price, strike_price, days_to_expiry, is_bullish)
            print(f"Contract selected: {option_symbol}, Strike: ${strike_price:.2f}, Price: ${option_price:.2f}")

            return option_symbol, option_price, strike_price
        except Exception as e:
            print(f"Error selecting option contract for {symbol}: {e}")
            return None, None, None

    def place_options_trade(self, symbol: str, price: float, direction: str, strike: float) -> bool:
        """Place an options trade (unchanged except for updated select_option_contract call)."""
        current_underlyings = self._get_current_underlyings()
        if len(current_underlyings) >= MAX_POSITIONS:
            print(f"Max positions ({MAX_POSITIONS}) reached. Skipping {symbol}.")
            return False
        try:
            is_bullish = direction == "CALL"
            contract_symbol, contract_price, strike_price = self.select_option_contract(symbol, price, is_bullish, strike)
            if not contract_symbol or not contract_price:
                print(f"Could not find suitable option contract for {symbol}")
                return False

            buying_power = self.get_buying_power()
            max_position_value = buying_power * POSITION_SIZE
            contracts = max(1, int(max_position_value / (contract_price * 100)))

            order_data = {
                'symbol': contract_symbol,
                'qty': contracts,
                'side': 'buy',
                'type': 'market',
                'time_in_force': 'day'
            }
            print("\n=== ORDER DETAILS ===")
            print(order_data)

            response = requests.post(f"{BASE_URL}/v2/orders", headers=HEADERS, json=order_data)
            if response.status_code == 200:
                print(f"""
                Placed {'bullish' if is_bullish else 'bearish'} options trade:
                Contract: {contract_symbol}
                Contracts: {contracts}
                Entry: ${contract_price:.2f}
                """)
                return True
            else:
                print(f"Error placing order: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error placing options trade for {symbol}: {e}")
            return False

    def get_buying_power(self) -> float:
        try:
            account = self.api.get_account()
            return float(account.buying_power)
        except Exception as e:
            print(f"Error fetching buying power: {e}")
            return 0.0

    def _get_current_underlyings(self) -> List[str]:
        try:
            positions = self.api.list_positions()
            underlyings = [re.match(r'^([A-Z]+)\d{6}[CP]\d{8}$', pos.symbol).group(1) for pos in positions if re.match(r'^([A-Z]+)\d{6}[CP]\d{8}$', pos.symbol)]
            return list(set(underlyings))
        except Exception as e:
            print(f"Error getting current underlyings: {e}")
            return []

    def run_trading_cycle(self):
        if not is_market_open(False):
            print("Market is closed. Skipping trading cycle.")
            return
        try:
            self.reset_daily_trades()
            print("\n=== Running Options Trading Cycle ===")
            filtered_stocks = get_filtered_stocks(100)
            if not filtered_stocks:
                print("No stocks meet criteria. Waiting for next cycle...")
                return

            current_underlyings = self._get_current_underlyings()
            for stock in filtered_stocks:
                symbol = stock['symbol']
                price = stock['price']
                direction = stock['direction']
                nearest_strike = stock['nearest_strike']
                if symbol in current_underlyings or symbol in self.traded_today or price < MIN_STOCK_PRICE:
                    print(f"Skipping {symbol}")
                    continue
                trade_success = self.place_options_trade(symbol, price, direction, nearest_strike)
                if trade_success:
                    self.traded_today.add(symbol)
        except Exception as e:
            print(f"Trading cycle error: {e}")

def is_market_open(skip: bool = False) -> bool:
    if skip:
        return True
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now <= close_time

def run_options_trading():
    executor = OptionsTradeExecutor()
    executor.run_trading_cycle()
    schedule.every(15).minutes.do(executor.run_trading_cycle)
    while True:
        try:
            schedule.run_pending()
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopping options trading...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_options_trading()