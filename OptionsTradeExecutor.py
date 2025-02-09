import re
import alpaca_trade_api as tradeapi
from typing import Dict, List, Tuple
import time
from datetime import datetime, timedelta
import pytz
import math

import schedule

from config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET
)
from optionsbotEnhancedVersion1 import(
    get_historical_data,
    calculate_technical_indicators,
    analyze_market_conditions,
    get_filtered_stocks
)
import requests  # Added this import
from datetime import datetime, timedelta
from scipy.stats import norm


def get_days_to_expiration():
    today = datetime.today()
    days_ahead = (4 - today.weekday()) % 7  # 4 = Friday (0 = Monday, 6 = Sunday)
    if days_ahead == 0:  # If today is Friday, move to next Friday
        days_ahead = 7
    return days_ahead
# Trading Parameters
POSITION_SIZE = 0.02  # 2% of portfolio per trade
MAX_POSITIONS = 5
PROFIT_TARGET = 20  # 50% profit target for options
STOP_LOSS = 20     # 30% stop loss for options
DAYS_TO_EXPIRATION = get_days_to_expiration()  # Target days to expiration
DELTA_TARGET = 0.30  # Target delta for options

# Stock Filtering Parameters
MIN_STOCK_PRICE = 5.0  # Minimum stock price
MIN_VOLUME = 1000000  # Minimum daily volume
MIN_MARKET_CAP = 1000000000  # Minimum market cap ($1B)


BASE_URL = 'https://paper-api.alpaca.markets'
DATA_URL = 'https://data.alpaca.markets'
HEADERS = {
    'APCA-API-KEY-ID': ALPACA_API_KEY,
    'APCA-API-SECRET-KEY': ALPACA_API_SECRET,
    'Content-Type': 'application/json'
}
class OptionsTradeExecutor:
    def __init__(self):
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL, api_version='v2')
        self.positions = []
        self.traded_today = set()
        self.last_trade_date = datetime.now(pytz.UTC).date()

    def reset_daily_trades(self):
        """Reset the daily traded symbols if a new day has started."""
        current_date = datetime.now(pytz.UTC).date()
        if current_date != self.last_trade_date:
            print("New day detected. Resetting traded symbols list.")
            self.traded_today.clear()
            self.last_trade_date = current_date

    def get_next_valid_expiry(self) -> datetime:
        """Get next valid expiration date (not today, next available Friday)"""
        today = datetime.now(pytz.UTC)
        days_ahead = (4 - today.weekday()) % 7  # 4 = Friday
        
        # If today is Friday, get next Friday
        if days_ahead == 0:
            days_ahead = 7
            
        # If it's too close to market close on Thursday, get next Friday
        if today.weekday() == 3 and today.hour >= 15:  # Thursday after 3 PM
            days_ahead = 8
            
        next_expiry = today + timedelta(days=days_ahead)
        return next_expiry.replace(hour=16, minute=0, second=0, microsecond=0)  # 4 PM ET

    def should_close_expiring_positions(self) -> bool:
        """Check if we should close positions expiring today"""
        now = datetime.now(pytz.UTC)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Close positions if it's the expiration day and within last hour of trading
        if now.weekday() == 4:  # Friday
            time_to_close = (market_close - now).total_seconds() / 3600
            return time_to_close <= 1
        return False

    def place_sell_order(self, symbol: str, qty: int):
        """Place a sell order to close an open position"""
        try:
            # Place the sell order for the option contract
            order_data = {
                'symbol': symbol,
                'qty': qty,
                'side': 'sell',
                'type': 'market',
                'time_in_force': 'day'
            }

            # Debugging: Print the request before sending
            print("\n=== SELL ORDER DETAILS ===")
            print(order_data)

            # Send the order request to sell the option
            response = requests.post(
                f"{BASE_URL}/v2/orders",
                headers=HEADERS,
                json=order_data
            )

            # Debugging: Check API response
            print(f"Sell Order Response: {response.status_code} - {response.text}")

            if response.status_code == 200:
                print(f"Sold {symbol} position with profit!")
            else:
                print(f"Error selling position: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error placing sell order for {symbol}: {e}")


    def close_expiring_positions(self):
        """Close all positions expiring today"""
        try:
            positions = self.api.list_positions()
            
            for position in positions:
                symbol = position.symbol
                
                # Extract expiration date from the option symbol (format: AAPL230915C150)
                expiry_str = symbol[-15:-9]  # Extract the 6-digit date (YYMMDD) from the symbol
                
                try:
                    expiry_date = datetime.strptime(expiry_str, '%y%m%d').date()
                except ValueError:
                    print(f"Error parsing expiration date for {symbol}")
                    continue  # Skip if the date is invalid
                
                # Check if the option expires today
                if expiry_date == datetime.now().date():
                    print(f"Closing position {symbol} due to same-day expiration")
                    self.place_sell_order(symbol, int(position.qty))
                    
        except Exception as e:
            print(f"Error closing expiring positions: {e}")


    def get_buying_power(self) -> float:
        """Get available buying power from account"""
        account = self.api.get_account()
        return float(account.buying_power)

    def find_nearest_strikes(self, current_price: float, num_strikes: int = 5) -> List[float]:
        """Find nearest strike prices based on current stock price"""
        # Round to nearest 0.5 for stocks under 100, nearest 1 for stocks over 100
        if current_price < 100:
            base_strike = round(current_price * 2) / 2
        else:
            base_strike = round(current_price)
            
        strikes = []
        for i in range(-num_strikes, num_strikes + 1):
            if current_price < 100:
                strike = base_strike + (i * 0.5)
            else:
                strike = base_strike + (i * 1.0)
            strikes.append(strike)
            
        return strikes

    def _get_current_underlyings(self) -> List[str]:
        """Get list of underlying symbols from current option positions"""
        positions = self.api.list_positions()
        underlyings = []
        for pos in positions:
            # Extract underlying from option symbol using regex
            match = re.match(r'^([A-Z]+)\d{6}[CP]\d{8}$', pos.symbol)
            if match:
                underlyings.append(match.group(1))
        return list(set(underlyings))  # Remove duplicates

    def monitor_positions(self):
        """Monitor existing positions and sell if the profit target is met"""
        try:
            # Get the current list of positions
            positions = self.api.list_positions()
            
            for position in positions:
                symbol = position.symbol
                qty = int(position.qty)  # Quantity of contracts
                avg_entry_price = float(position.avg_entry_price)  # Entry price
                current_price = float(position.current_price)  # Current price of the option contract
                
                # Calculate current profit/loss percentage
                profit_loss_percent = ((current_price - avg_entry_price) / avg_entry_price) * 100
                
                print(f"\nMonitoring {symbol}:")
                print(f"Contracts: {qty}")
                print(f"Avg Entry Price: ${avg_entry_price:.2f}")
                print(f"Current Price: ${current_price:.2f}")
                print(f"P&L: ${float(position.unrealized_pl):.2f} ({profit_loss_percent:.2f}%)")
                
                # Sell position if profit is 20% or higher
                if profit_loss_percent <= -STOP_LOSS or profit_loss_percent >= PROFIT_TARGET:
                    print(f"Profit target reached for {symbol}. Selling position.")
                    self.place_sell_order(symbol, qty)
                    
        except Exception as e:
            print(f"Error monitoring positions: {e}")


    def calculate_option_price(self, stock_price: float, strike_price: float,
                               days_to_expiry: int, is_call: bool,
                               volatility: float = 0.3, risk_free_rate: float = 0.01) -> float:
        """
        Calculate the theoretical option price using the Black–Scholes formula.

        Parameters:
          stock_price (float): Current price of the underlying asset (S).
          strike_price (float): Option strike price (K).
          days_to_expiry (int): Days to expiration; converted to years (T).
          is_call (bool): True for call option, False for put option.
          volatility (float): Annualized volatility (σ). Default is 0.3.
          risk_free_rate (float): Annualized risk-free rate (r). Default is 0.01 (1%).

        Returns:
          float: Option price rounded to 2 decimal places.
        """
        T = days_to_expiry / 365.0
        # If expiration has passed, return intrinsic value
        if T <= 0:
            return max(0, (stock_price - strike_price) if is_call else (strike_price - stock_price))

        S = stock_price
        K = strike_price
        sigma = volatility
        r = risk_free_rate

        # Compute d1 and d2 as per Black-Scholes
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if is_call:
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return round(price, 2)

    def place_options_trade(self, symbol: str, price: float, direction: str) -> bool:
        """Place options trade based on market conditions"""
        try:
            # Determine trade direction based on market conditions
            if direction == "CALL":
                is_bullish = True 
            else:
                is_bullish = False            
            
            # Select appropriate option contract
            contract_symbol, contract_price, strike_price = self.select_option_contract(
                symbol, 
                price, 
                is_bullish
            )
            
            if not contract_symbol or not contract_price:
                print(f"Could not find suitable option contract for {symbol}")
                return False

            # Calculate position size
            buying_power = self.get_buying_power()
            max_position_value = buying_power * POSITION_SIZE
            contracts = max(1, int(max_position_value / (contract_price * 100)))
            
            # Prepare order data (Simple Limit Order without bracket)
            order_data = {
                'symbol': contract_symbol,
                'qty': contracts,
                'side': 'buy',
                'type': 'market',
                'time_in_force': 'day'
            }

            # Debugging: Print the request before sending
            print("\n=== ORDER DETAILS ===")
            print(order_data)

            # Send order request (this is a simple limit order, no bracket)
            response = requests.post(
                f"{BASE_URL}/v2/orders",
                headers=HEADERS,
                json=order_data
            )

            # Debugging: Check API response
            print(f"Order Response: {response.status_code} - {response.text}")

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


    def select_option_contract(self, symbol: str, current_price: float, is_bullish: bool) -> Tuple[str, float, float]:
        """Select appropriate option contract with valid expiration"""
        try:
            # Get next valid expiration
            next_expiry = self.get_next_valid_expiry()
            
            # Get nearest strikes
            strikes = self.find_nearest_strikes(current_price)
            
            # Select strike based on direction
            if is_bullish:
                strike = max([s for s in strikes if s <= current_price])
            else:
                strike = min([s for s in strikes if s >= current_price])
            
            # Calculate theoretical option price
            option_price = self.calculate_option_price(
                current_price, 
                strike, 
                (next_expiry - datetime.now(pytz.UTC)).days,
                is_bullish
            )
            
            # Generate option symbol with next valid expiry
            expiry_str = next_expiry.strftime('%y%m%d')
            option_type = 'C' if is_bullish else 'P'
            strike_str = str(int(strike * 1000)).zfill(8)
            option_symbol = f"{symbol}{expiry_str}{option_type}{strike_str}"
            
            return option_symbol, option_price, strike
            
        except Exception as e:
            print(f"Error selecting option contract for {symbol}: {e}")
            return None, None, None

    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            self.reset_daily_trades()  # Reset if a new day has started
            print("\n=== Running Options Trading Cycle ===")
            
            # First check if we need to close any expiring positions
            if self.should_close_expiring_positions():
                print("Closing positions expiring today...")
                self.close_expiring_positions()
            
            # Get and process new trading opportunities
            filtered_stocks = get_filtered_stocks(100)
            
            for stock in filtered_stocks:
                if not filtered_stocks:
                    print("No stocks meet criteria. Waiting for next cycle....")
                    continue
                    
                symbol = stock['symbol']
                price = stock['price']
                direction = stock['direction']
                
                current_underlyings = self._get_current_underlyings()
                if symbol in current_underlyings:
                    print(f"⚠️ Skipping {symbol} - open position exists")
                    continue
                if symbol in self.traded_today:
                    print(f"⚠️ Skipping {symbol} - already traded today")
                    continue

                if price < MIN_STOCK_PRICE:
                    print(f"Skipping {symbol}: Price (${price:.2f}) below minimum (${MIN_STOCK_PRICE:.2f})")
                    continue
                # Place new trades
                trade_success = self.place_options_trade(symbol, price, direction)
                if trade_success:
                    # Record that this underlying has been traded today
                    self.traded_today.add(symbol)
        except Exception as e:
            print(f"Trading cycle error: {e}")

def run_options_trading():
    """Main trading loop with proper error handling and scheduling"""
    executor = OptionsTradeExecutor()
    executor.run_trading_cycle()
    schedule.every(30).minutes.do(executor.run_trading_cycle)
    schedule.every(1).minutes.do(executor.monitor_positions)
    while True:
        try:
            schedule.run_pending()
            #print("\nWaiting for next cycle...")
            time.sleep(5)  # 5-secs delay between cycles
            
        except KeyboardInterrupt:
            print("\nStopping options trading...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_options_trading()