import asyncio
import re
import alpaca_trade_api as tradeapi
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime, timedelta, date
import pytz
import os
import csv
import requests
from scipy.stats import norm
import math
import schedule
import pandas as pd
import numpy as np

from alerts import send_telegram_alert
from analysis import analyze_market_conditions_day_trading
from config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET
)
from optionsbotEnhancedVersion1 import(
    calculate_technical_indicators,
    get_filtered_stocks,
    get_historical_data,
)

# =========== CONFIGURATION PARAMETERS (Easy to modify) ===========

# File paths
TRADING_JOURNAL_FILE = "options_trading_journal.csv"  # CSV journal file path

# Position sizing and risk management
POSITION_SIZE_PERCENT = 0.01  # Use 1% of buying power per position
MAX_POSITIONS = 10          # Maximum number of positions to hold
STOP_LOSS_PERCENT = 20     # Cut losses at -20%
MAX_PROFIT_PERCENT = 30

# Stock filtering
MIN_STOCK_PRICE = 20.0     # Minimum stock price $20

# Trading schedule
SCAN_INTERVAL_MINUTES = 15  # Run scan every X minutes
MONITOR_INTERVAL_MINUTES = 2  # Monitor positions every X minutes

# Technical analysis parameters - UPDATED for more permissive entry
ENTRY_TREND_THRESHOLD = 1    # Reduced from 2 to 1
ENTRY_MOMENTUM_THRESHOLD = 1 # Reduced from 2 to 1
ENTRY_RISK_THRESHOLD = 0     # Reduced from 1 to 0
OTM_PERCENT_CALLS = 1.03  # For calls, use X% OTM (3% here)
OTM_PERCENT_PUTS = 0.97   # For puts, use X% OTM (3% here)
TREND_REVERSAL_DAYS = 3   # Number of days to check for trend reversal

# API Configuration
BASE_URL = 'https://paper-api.alpaca.markets'
DATA_URL = 'https://data.alpaca.markets'
HEADERS = {
    'APCA-API-KEY-ID': ALPACA_API_KEY,
    'APCA-API-SECRET-KEY': ALPACA_API_SECRET,
    'Content-Type': 'application/json'
}

# ===============================================================

class OptionsTradeExecutor:
    def __init__(self) :
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL, api_version='v2')
        self.traded_today = set()
        self.last_trade_date = datetime.now(pytz.UTC).date()
        
        # Position tracking
        self.position_metrics = {}
        
        # Store filtered stocks for reference
        self.filtered_stocks = []
        
        # Initialize trading journal if it doesn't exist
        self.initialize_trading_journal()

    def initialize_trading_journal(self):
        """Initialize the CSV trading journal if it doesn't exist"""
        if not os.path.exists(TRADING_JOURNAL_FILE):
            with open(TRADING_JOURNAL_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Date', 
                    'Time', 
                    'Symbol', 
                    'Underlying', 
                    'Direction', 
                    'Action', 
                    'Contracts',
                    'Price', 
                    'Total_Value',
                    'Expiration', 
                    'Strike',
                    'Exit_Reason',
                    'P_L_Amount',
                    'P_L_Percent',
                    'Trade_Duration',
                    'Notes'
                ])
            print(f"Created new trading journal file: {TRADING_JOURNAL_FILE}")

    def log_trade_to_journal(self, symbol, action, contracts, price, 
                            exit_reason=None, entry_price=None, entry_time=None, 
                            p_l_amount=None, p_l_percent=None, notes=None):
        """Enhanced trade logging with better symbol parsing and error handling"""
        try:
            # Extract underlying symbol with improved regex pattern
            underlying = symbol
            # Try different regex patterns for different option symbol formats
            patterns = [
                r'^([A-Z]+)\d{6}[CP]\d{8}$',  # Format: AAPL230915C00150000
                r'^([A-Z]+)\d{6}[CP]\d+$',    # Format: AAPL230915C150
                r'^([A-Z]+)_\d+$'             # Format: AAPL_150
            ]
            
            for pattern in patterns:
                match = re.match(pattern, symbol)
                if match:
                    underlying = match.group(1)
                    break
            
            # Determine direction with better parsing
            if 'C' in symbol:
                direction = "CALL"
            elif 'P' in symbol:
                direction = "PUT"
            else:
                direction = "STOCK"
            
            # Calculate total value
            total_value = contracts * price * 100
            
            # Get current date and time
            now = datetime.now(pytz.UTC)
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            # Calculate trade duration if applicable
            trade_duration = None
            if action == 'SELL' and entry_time:
                trade_duration = (now - entry_time).total_seconds() / 3600
            
            # Extract expiration and strike if possible
            expiration = "N/A"
            strike = "0"
            
            for pattern, exp_group, strike_group in [
                (r'^[A-Z]+(\d{6})[CP](\d{8})$', 1, 2),  # AAPL230915C00150000
                (r'^[A-Z]+(\d{6})[CP](\d+)$', 1, 2)     # AAPL230915C150
            ]:
                match = re.match(pattern, symbol)
                if match:
                    try:
                        exp_str = match.group(exp_group)
                        exp_date = datetime.strptime(exp_str, '%y%m%d').date()
                        expiration = exp_date.strftime('%Y-%m-%d')
                        
                        strike_str = match.group(strike_group)
                        if len(strike_str) == 8:  # Format with padding zeros
                            strike = str(int(strike_str) / 1000)
                        else:
                            strike = strike_str
                    except Exception as e:
                        print(f"Error parsing expiration or strike: {e}")
            
            # Append to CSV
            with open(TRADING_JOURNAL_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    date_str, time_str, symbol, underlying, direction, action, contracts,
                    f"{price:.2f}", f"{total_value:.2f}", expiration, strike,
                    exit_reason or '', 
                    f"{p_l_amount:.2f}" if p_l_amount is not None else '',
                    f"{p_l_percent:.2f}" if p_l_percent is not None else '',
                    f"{trade_duration:.2f}" if trade_duration else '',
                    notes or ''
                ])
            
            print(f"Trade logged to journal: {underlying}, {action}, {contracts} contracts at ${price:.2f}")
            return True
        
        except Exception as e:
            print(f"Error logging trade to journal: {e}")
            # Try a simplified fallback logging
            try:
                with open(TRADING_JOURNAL_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now(pytz.UTC).strftime("%Y-%m-%d"),
                        datetime.now(pytz.UTC).strftime("%H:%M:%S"),
                        symbol, "UNKNOWN", "UNKNOWN", action, contracts,
                        f"{price:.2f}", f"{contracts * price * 100:.2f}", "N/A", "0",
                        exit_reason or '', '', '', '', f"Error in logging: {e}"
                    ])
                return True
            except:
                return False

    def reset_daily_trades(self):
        """Reset the daily traded symbols if a new day has started."""
        current_date = datetime.now(pytz.UTC).date()
        if current_date != self.last_trade_date:
            print("New day detected. Resetting traded symbols list.")
            self.traded_today.clear()
            self.last_trade_date = current_date


    def should_close_expiring_positions(self) -> bool:
        """Check if we should close positions expiring today"""
        now = datetime.now(pytz.UTC)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Close positions if it's the expiration day and within 30 mins of market close
        if now.weekday() == 4:  # Friday
            time_to_close = (market_close - now).total_seconds() / 60
            return time_to_close <= 30
        return False

    def place_sell_order(self, symbol: str, qty: int, is_call: bool, entry_price: float, entry_time=None, exit_reason=None):
        """Place a sell order to close an open position with journal logging"""
        try:
            # Place the sell order for the option contract
            order_data = {
                'symbol': symbol,
                'qty': qty,
                'side': 'sell',
                'type': 'market',
                'time_in_force': 'day'
            }

            # Print the request before sending
            print("\n=== SELL ORDER DETAILS ===")
            print(order_data)

            # Add retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Send the order request to sell the option
                    response = requests.post(
                        f"{BASE_URL}/v2/orders",
                        headers=HEADERS,
                        json=order_data,
                        timeout=10  # Add timeout
                    )
                    
                    # Check API response
                    print(f"Sell Order Response: {response.status_code} - {response.text}")
                    
                    if response.status_code == 200:
                        print(f"Sold {symbol} position!")
                        
                        # Get current market price for tracking performance
                        try:
                            current_price = float(self.api.get_latest_trade(symbol).price)
                            
                            # Calculate P&L metrics
                            p_l_amount = (current_price - entry_price) * qty * 100  # Each contract is 100 shares
                            p_l_percent = ((current_price - entry_price) / entry_price) * 100
                            
                            # Determine if it's a win or loss
                            emoji = "🟢 PROFIT" if p_l_percent > 0 else "🔴 LOSS"
                            
                            # Parse the option symbol to get the underlying ticker
                            # Typical format: SPY250311P00540000 (Symbol, Date, Put/Call, Strike)
                            underlying = symbol
                            for pattern in [r'^([A-Z]+)\d', r'^([A-Z]+)_']:
                                match = re.match(pattern, symbol)
                                if match:
                                    underlying = match.group(1)
                                    break
                            
                            # Create telegram message with detailed P&L
                            message = f"""{emoji} - POSITION CLOSED:
                            Symbol: {underlying}
                            Contract: {symbol}
                            Strategy: {'CALL' if is_call else 'PUT'}
                            Exit Price: ${current_price:.2f}
                            Entry Price: ${entry_price:.2f}
                            Contracts: {qty}
                            P&L Amount: ${p_l_amount:.2f}
                            P&L Percent: {p_l_percent:.2f}%
                            Exit Reason: {exit_reason if exit_reason else 'Manual exit'}
                            {'Trade Duration: ' + f"{((datetime.now(pytz.UTC) - entry_time).total_seconds() / 3600):.2f} hours" if entry_time else ''}
                            """
                            
                            # Send alert with proper data
                            try:
                                alert_data = {
                                    'symbol': underlying,
                                    'action': 'SELL',
                                    'price': current_price,
                                    'quantity': qty,
                                    'p_l': p_l_amount,
                                    'p_l_percent': p_l_percent,
                                    'exit_reason': exit_reason,
                                    'message': message
                                }
                                asyncio.run(send_telegram_alert(alert_data))
                            except Exception as alert_error:
                                print(f"Error sending alert: {alert_error}")
                            
                            # Log the trade to the journal using only the underlying symbol
                            try:
                                self.log_trade_to_journal(
                                    symbol=symbol,  # Use the full option symbol for proper parsing
                                    action='SELL',
                                    contracts=qty,
                                    price=current_price,
                                    exit_reason=exit_reason,
                                    entry_price=entry_price,
                                    entry_time=entry_time,
                                    p_l_amount=p_l_amount,
                                    p_l_percent=p_l_percent,
                                    notes=f"Option {symbol} closed: {exit_reason}"
                                )
                                print(f"Trade successfully logged to journal for {underlying}")
                            except Exception as journal_error:
                                print(f"Journal logging error: {journal_error}")
                                
                        except Exception as e:
                            print(f"Error updating trade journal: {e}")
                        
                        return True
                    
                    # Handle specific error cases
                    elif response.status_code == 429:  # Rate limit
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"Rate limited. Waiting {wait_time} seconds before retry.")
                        time.sleep(wait_time)
                        continue
                    
                    elif attempt < max_retries - 1:
                        print(f"Order failed. Retrying ({attempt+1}/{max_retries})...")
                        time.sleep(1)
                        continue
                    
                    else:
                        print(f"Order failed after {max_retries} attempts.")
                        return False
                
                except Exception as e:
                    print(f"Error placing sell order: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying ({attempt+1}/{max_retries})...")
                        time.sleep(1)
                    else:
                        return False

            return False

        except Exception as e:
            print(f"Error placing sell order for {symbol}: {e}")
            return False

    def close_expiring_positions(self):
        """Close all positions expiring today"""
        try:
            positions = self.api.list_positions()
            
            for position in positions:
                symbol = position.symbol
                
                # Extract expiration date from the option symbol with improved parsing
                expiry_date = None
                for pattern, exp_group in [
                    (r'^[A-Z]+(\d{6})[CP]\d{8}$', 1),  # AAPL230915C00150000
                    (r'^[A-Z]+(\d{6})[CP]\d+$', 1)     # AAPL230915C150
                ]:
                    match = re.match(pattern, symbol)
                    if match:
                        try:
                            exp_str = match.group(exp_group)
                            expiry_date = datetime.strptime(exp_str, '%y%m%d').date()
                            break
                        except ValueError:
                            print(f"Error parsing expiration date for {symbol}")
                
                if not expiry_date:
                    print(f"Could not parse symbol format for {symbol}")
                    continue  # Skip if pattern doesn't match
                
                # Check if the option expires today
                if expiry_date == datetime.now().date():
                    print(f"Closing position {symbol} due to same-day expiration")
                    # Extract is_call from the symbol
                    is_call = 'C' in symbol
                    entry_price = float(position.avg_entry_price)
                    
                    # Get position metric to find entry time if available
                    position_key = f"{symbol}_{int(position.qty)}_{entry_price}"
                    entry_time = None
                    if position_key in self.position_metrics and 'entry_time' in self.position_metrics[position_key]:
                        entry_time = self.position_metrics[position_key]['entry_time']
                    
                    self.place_sell_order(
                        symbol, 
                        int(position.qty), 
                        is_call, 
                        entry_price, 
                        entry_time,
                        exit_reason="Expiration day"
                    )
                    
        except Exception as e:
            print(f"Error closing expiring positions: {e}")

    def get_buying_power(self) -> float:
        """Get available buying power from account"""
        account = self.api.get_account()
        return float(account.buying_power)

    def _get_current_underlyings(self) -> List[str]:
        """Get list of underlying symbols from current option positions with improved parsing"""
        positions = self.api.list_positions()
        underlyings = []
        
        for pos in positions:
            # Extract underlying with multiple pattern matching
            underlying = pos.symbol  # Default to full symbol if no match
            
            for pattern in [r'^([A-Z]+)\d{6}[CP]\d{8}$', r'^([A-Z]+)\d{6}[CP]\d+$', r'^([A-Z]+)_\d+$']:
                match = re.match(pattern, pos.symbol)
                if match:
                    underlying = match.group(1)
                    break
                    
            underlyings.append(underlying)
            
        return list(set(underlyings))  # Remove duplicates

    def monitor_positions(self):
        """Monitor positions with price-based and technical exit signals."""
        try:
            self.check_expiring_options()
            positions = self.api.list_positions()
            
            for position in positions:
                symbol = position.symbol
                qty = int(position.qty)
                avg_entry_price = float(position.avg_entry_price)
                current_price = float(position.current_price)
                is_call = 'C' in symbol[-9:-8]
                profit_loss_percent = ((current_price - avg_entry_price) / avg_entry_price) * 100
                
                print(f"\nMonitoring {symbol}:")
                print(f"Contracts: {qty}, Entry: ${avg_entry_price:.2f}, Current: ${current_price:.2f}")
                print(f"P&L: ${float(position.unrealized_pl):.2f} ({profit_loss_percent:.2f}%)")
                
                # Extract underlying symbol with improved parsing
                underlying = symbol  # Default to full symbol if no match
                
                for pattern in [r'^([A-Z]+)\d{6}[CP]\d{8}$', r'^([A-Z]+)\d{6}[CP]\d+$', r'^([A-Z]+)_\d+$']:
                    match = re.match(pattern, symbol)
                    if match:
                        underlying = match.group(1)
                        break
                
                if not underlying or underlying == symbol:
                    print(f"Could not extract underlying from {symbol}, skipping analysis")
                    continue
                
                print(f"\nAnalyzing {symbol}...")
                bars = get_historical_data(underlying)
                if bars is None:
                    continue
                df = calculate_technical_indicators(bars)
                if df is None:
                    continue

                # Call the updated analysis function
                analysis = analyze_market_conditions_day_trading(df, None)
                
                # Check exit conditions
                if 'exit_signals' in analysis and analysis['exit_signals']:
                    exit_reason = ', '.join(analysis['exit_signals'])
                    self._exit_position(symbol, qty, is_call, avg_entry_price, f"Technical: {exit_reason}", profit_loss_percent)
                elif profit_loss_percent <= -STOP_LOSS_PERCENT:
                    self._exit_position(symbol, qty, is_call, avg_entry_price, "Stop-loss", profit_loss_percent)
                elif profit_loss_percent >= MAX_PROFIT_PERCENT:
                    self._exit_position(symbol, qty, is_call, avg_entry_price, "Profit target", profit_loss_percent)
                        
        except Exception as e:
            print(f"Error monitoring positions: {e}")

    def _exit_position(self, symbol, qty, is_call, entry_price, exit_reason, profit_loss_percent):
        """Helper method to exit positions"""
        try:
            message = f"Exit signal triggered for {symbol}: {exit_reason} ({profit_loss_percent:.2f}%). Selling position."
            print(message)
            
            # Get position metric to find entry time if available
            position_key = f"{symbol}_{qty}_{entry_price}"
            entry_time = None
            if position_key in self.position_metrics and 'entry_time' in self.position_metrics[position_key]:
                entry_time = self.position_metrics[position_key]['entry_time']
            
            # Sell the position
            self.place_sell_order(
                symbol=symbol, 
                qty=qty, 
                is_call=is_call, 
                entry_price=entry_price, 
                entry_time=entry_time, 
                exit_reason=exit_reason
            )
        except Exception as e:
            print(f"Error exiting position: {e}")
    
    def get_option_contracts(self, symbol: str) -> List[Dict]:
        """Enhanced option contract retrieval with better error handling and pagination"""
        all_contracts = []
        try:
            url = "https://paper-api.alpaca.markets/v2/options/contracts"
            params = {
                'underlying_symbols': symbol,
                'status': 'active',
                'limit': 1000
            }
            
            # Add retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = requests.get(url, headers=HEADERS, params=params, timeout=10) 
                    if response.status_code == 200:
                        data = response.json()
                        contracts = data.get('option_contracts', [])
                        all_contracts.extend(contracts)
                        
                        # Check if we need to paginate
                        next_page_token = data.get('next_page_token')
                        if not next_page_token:
                            break
                            
                        params['page_token'] = next_page_token
                    else:
                        print(f"Error fetching option contracts for {symbol}: {response.status_code} - {response.text}")
                        retry_count += 1
                        time.sleep(2)  # Wait before retrying
                except Exception as e:
                    print(f"Exception in API call: {e}")
                    retry_count += 1
                    time.sleep(2)  # Wait before retrying
            
            call_count = sum(1 for c in all_contracts if c.get('type', '').lower() == 'call')
            put_count = sum(1 for c in all_contracts if c.get('type', '').lower() == 'put')
            print(f"Fetched {len(all_contracts)} option contracts for {symbol} - {call_count} calls, {put_count} puts")
            
            # Validate contract data
            valid_contracts = []
            for contract in all_contracts:
                if all(k in contract for k in ['symbol', 'type', 'strike_price', 'expiration_date']):
                    valid_contracts.append(contract)
                else:
                    print(f"Skipping invalid contract: {contract}")
            
            print(f"Valid contracts: {len(valid_contracts)} out of {len(all_contracts)}")
            return valid_contracts
            
        except Exception as e:
            print(f"Exception fetching option contracts for {symbol}: {e}")
            return []

    def select_option_contract(self, symbol: str, price: float, is_bullish: bool) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """Enhanced option contract selection with better date handling and error recovery"""
        try:
            desired_type = 'call' if is_bullish else 'put'
            print(f"Selecting {desired_type} option for {symbol} at ${price:.2f}")
            
            # Get all option contracts from API
            contracts = self.get_option_contracts(symbol)
            if not contracts:
                print(f"No option contracts found for {symbol}")
                return None, None, None
            
            # Filter by option type
            type_filtered_contracts = [c for c in contracts if c.get('type', '').lower() == desired_type]
            print(f"Found {len(type_filtered_contracts)} {desired_type} contracts")
            
            if not type_filtered_contracts:
                print(f"No {desired_type} contracts found for {symbol}")
                return None, None, None
            
            # Avoid same-day expiration
            today = datetime.now(pytz.UTC).date()
            
            # Sort contracts by expiration date (ascending)
            sorted_contracts = sorted(
                type_filtered_contracts,
                key=lambda c: datetime.strptime(c.get('expiration_date', '2099-12-31'), '%Y-%m-%d').date()
            )
            
            # Group contracts by expiration date
            expiry_groups = {}
            for contract in sorted_contracts:
                try:
                    exp_date = datetime.strptime(contract.get('expiration_date', ''), '%Y-%m-%d').date()
                    if exp_date > today:  # Skip contracts expiring today
                        if exp_date not in expiry_groups:
                            expiry_groups[exp_date] = []
                        expiry_groups[exp_date].append(contract)
                except (ValueError, TypeError) as e:
                    print(f"Error parsing date {contract.get('expiration_date')}: {e}")
                    continue
            
            if not expiry_groups:
                print(f"No valid expiration dates found for {symbol}")
                return None, None, None
            
            # Select the nearest expiration date (but not today)
            nearest_expiry = min(expiry_groups.keys())
            contracts_for_expiry = expiry_groups[nearest_expiry]
            
            print(f"Selected expiration date: {nearest_expiry}, with {len(contracts_for_expiry)} contracts")
            
            # Calculate target strike price
            target_strike = price * (OTM_PERCENT_CALLS if is_bullish else OTM_PERCENT_PUTS)
            
            # Find closest strike from selected expiry
            best_contract = min(contracts_for_expiry, 
                            key=lambda c: abs(float(c.get('strike_price', 0)) - target_strike))
            
            contract_symbol = best_contract.get('symbol', '')
            strike = float(best_contract.get('strike_price', 0))
            expiration_date = best_contract.get('expiration_date', '')
            
            print(f"Selected {desired_type} option {contract_symbol} with strike ${strike:.2f} expiring {expiration_date}")
            
            # Get contract price with fallbacks
            contract_price = None
            for price_field in ['ask_price', 'last_price', 'mark_price', 'close_price']:
                if price_field in best_contract and best_contract[price_field]:
                    try:
                        contract_price = float(best_contract[price_field])
                        print(f"Using {price_field}: ${contract_price:.2f}")
                        break
                    except (ValueError, TypeError):
                        continue
            
            # If no price found, use a default
            if contract_price is None:
                contract_price = 1.0
                print(f"No valid price found, using default: ${contract_price:.2f}")
            
            return contract_symbol, contract_price, strike
        
        except Exception as e:
            print(f"Error selecting option contract for {symbol}: {e}")
            return None, None, None

    def calculate_position_size(self, contract_price: float, risk_level: int = 1) -> int:
        """Enhanced position sizing with risk level adjustment"""
        try:
            # Get account equity
            account = self.api.get_account()
            equity = float(account.equity)
            
            # Base position size on account percentage
            base_percentage = POSITION_SIZE_PERCENT
            
            # Adjust based on risk level
            if risk_level >= 3:
                risk_multiplier = 1.0  # Full size for high conviction trades
            elif risk_level == 2:
                risk_multiplier = 0.8  # 80% size for medium conviction
            elif risk_level == 1:
                risk_multiplier = 0.6  # 60% size for lower conviction
            else:
                risk_multiplier = 0.4  # 40% size for minimal conviction
            
            # Calculate position value
            max_position_value = equity * base_percentage * risk_multiplier
            
            # Calculate number of contracts
            contract_cost = contract_price * 100
            contracts = int(max_position_value / contract_cost)
            
            # Minimum of 1 contract, maximum of 100
            contracts = max(1, min(100, contracts))
            
            print(f"Position sizing: {contracts} contracts at ${contract_price:.2f}/contract")
            print(f"Total position value: ${contracts * contract_price * 100:.2f}")
            print(f"Risk level: {risk_level}, Risk multiplier: {risk_multiplier}")
            
            return contracts
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 1  # Default to 1 contract
        
    def place_options_trade(self, symbol: str, price: float, direction: str) -> bool:
        """Enhanced options trade placement with better error handling and retry logic"""
        try:
            # Determine trade direction
            is_bullish = direction == "CALL"
            
            # Get risk level from filtered stocks data
            risk_level = 1  # Default
            for stock in self.filtered_stocks:
                if stock['symbol'] == symbol:
                    risk_level = stock.get('risk_level', 1)
                    break
            
            # Select appropriate option contract
            contract_symbol, contract_price, strike_price = self.select_option_contract(
                symbol, 
                price, 
                is_bullish
            )
            
            if not contract_symbol or not contract_price:
                print(f"Could not find suitable option contract for {symbol}")
                return False
            
            # Validate expiration date
            match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d+)$', contract_symbol)
            if match:
                expiry_str = match.group(2)  # YYMMDD
                today = datetime.now().strftime('%y%m%d')
                
                if expiry_str == today:
                    print(f"Skipping {contract_symbol} - expires today")
                    return False
            
            # Calculate position size with risk level
            contracts = self.calculate_position_size(contract_price, risk_level)
            
            # Place order
            order_data = {
                'symbol': contract_symbol,
                'qty': contracts,
                'side': 'buy',
                'type': 'market',
                'time_in_force': 'day'
            }

            print("\n=== ORDER DETAILS ===")
            print(order_data)

            # Add retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{BASE_URL}/v2/orders",
                        headers=HEADERS,
                        json=order_data,
                        timeout=10  # Add timeout
                    )
                    
                    print(f"Order Response: {response.status_code} - {response.text}")
                    
                    if response.status_code == 200:
                        # Store entry time for position tracking
                        position_key = f"{contract_symbol}_{contracts}_{contract_price}"
                        self.position_metrics[position_key] = {
                            'entry_time': datetime.now(pytz.UTC),
                            'entry_price': contract_price
                        }
                        
                        # Send alert
                        try:
                            alert_data = {
                                'symbol': symbol,
                                'action': 'BUY',
                                'price': contract_price,
                                'quantity': contracts,
                                'direction': direction
                            }
                            asyncio.run(send_telegram_alert(alert_data))
                        except Exception as alert_error:
                            print(f"Error sending alert: {alert_error}")
                        
                        # Log the trade
                        try:
                            self.log_trade_to_journal(
                                symbol=contract_symbol,
                                action='BUY',
                                contracts=contracts,
                                price=contract_price,
                                notes=f"New position opened for {symbol}"
                            )
                        except Exception as log_error:
                            print(f"Error logging trade: {log_error}")
                        
                        return True
                    
                    # Handle specific error cases
                    elif response.status_code == 403 and "insufficient" in response.text.lower() and contracts > 1:
                        # Try with fewer contracts
                        reduced_contracts = max(1, contracts // 2)
                        print(f"Retrying with reduced size: {reduced_contracts} contracts")
                        
                        order_data['qty'] = reduced_contracts
                        continue  # Try again with reduced size
                    
                    elif response.status_code == 429:  # Rate limit
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"Rate limited. Waiting {wait_time} seconds before retry.")
                        time.sleep(wait_time)
                        continue
                    
                    elif attempt < max_retries - 1:
                        print(f"Order failed. Retrying ({attempt+1}/{max_retries})...")
                        time.sleep(1)
                        continue
                    
                    else:
                        print(f"Order failed after {max_retries} attempts.")
                        return False
                    
                except Exception as e:
                    print(f"Error placing order: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying ({attempt+1}/{max_retries})...")
                        time.sleep(1)
                    else:
                        return False
                
            return False
                
        except Exception as e:
            print(f"Error placing options trade: {e}")
            return False

    def check_expiring_options(self):
        """Check for and potentially exit same-day expiring options with improved date parsing"""
        try:
            positions = self.api.list_positions()
            today = datetime.now(pytz.UTC).date()
            
            for position in positions:
                symbol = position.symbol
                
                # Extract expiration date with multiple pattern matching
                expiry_date = None
                
                for pattern, exp_group in [
                    (r'^[A-Z]+(\d{6})[CP]\d{8}$', 1),  # AAPL230915C00150000
                    (r'^[A-Z]+(\d{6})[CP]\d+$', 1)     # AAPL230915C150
                ]:
                    match = re.match(pattern, symbol)
                    if match:
                        try:
                            exp_str = match.group(exp_group)
                            year = int('20' + exp_str[:2])
                            month = int(exp_str[2:4])
                            day = int(exp_str[4:6])
                            expiry_date = date(year, month, day)
                            break
                        except (ValueError, TypeError) as e:
                            print(f"Error parsing date from {exp_str}: {e}")
                
                if not expiry_date:
                    print(f"Could not parse expiration date for {symbol}")
                    continue
                    
                # If expires today, exit the position
                if expiry_date == today:
                    print(f"Closing position {symbol} due to same-day expiration")
                    is_call = 'C' in symbol
                    qty = int(position.qty)
                    entry_price = float(position.avg_entry_price)
                    
                    # Get position metric to find entry time if available
                    position_key = f"{symbol}_{qty}_{entry_price}"
                    entry_time = None
                    if position_key in self.position_metrics and 'entry_time' in self.position_metrics[position_key]:
                        entry_time = self.position_metrics[position_key]['entry_time']
                    
                    self._exit_position(
                        symbol=symbol,
                        qty=qty,
                        is_call=is_call,
                        entry_price=entry_price,
                        exit_reason="same-day expiration",
                        profit_loss_percent=0  # placeholder, not important for this case
                    )
                    
        except Exception as e:
            print(f"Error checking expiring options: {e}")

    def analyze_trading_performance(self):
        """Analyze trading performance from journal data"""
        try:
            if not os.path.exists(TRADING_JOURNAL_FILE):
                print("Trading journal file not found.")
                return
                
            # Read trading journal
            df = pd.read_csv(TRADING_JOURNAL_FILE)
            
            if df.empty:
                print("No trades recorded yet.")
                return
                
            # Convert date columns
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Calculate basic statistics
            total_trades = len(df[df['Action'] == 'SELL'])
            if total_trades == 0:
                print("No closed trades found.")
                return
                
            # Filter to only closed trades
            closed_trades = df[df['Action'] == 'SELL'].copy()
            
            # Convert P/L columns to numeric
            closed_trades['P_L_Amount'] = pd.to_numeric(closed_trades['P_L_Amount'], errors='coerce')
            closed_trades['P_L_Percent'] = pd.to_numeric(closed_trades['P_L_Percent'], errors='coerce')
            
            # Calculate win rate
            winning_trades = closed_trades[closed_trades['P_L_Amount'] > 0]
            win_rate = len(winning_trades) / total_trades * 100
            
            # Calculate average P/L
            avg_pl_amount = closed_trades['P_L_Amount'].mean()
            avg_pl_percent = closed_trades['P_L_Percent'].mean()
            
            # Calculate max drawdown
            cumulative_pl = closed_trades['P_L_Amount'].cumsum()
            max_drawdown = (cumulative_pl.cummax() - cumulative_pl).max()
            
            # Calculate Sharpe ratio (simplified)
            if closed_trades['P_L_Percent'].std() > 0:
                sharpe_ratio = (closed_trades['P_L_Percent'].mean() / closed_trades['P_L_Percent'].std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0
                
            # Print performance summary
            print("\n=== TRADING PERFORMANCE SUMMARY ===")
            print(f"Total Closed Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Average P/L: ${avg_pl_amount:.2f} ({avg_pl_percent:.2f}%)")
            print(f"Max Drawdown: ${max_drawdown:.2f}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            
            # Analyze by underlying
            by_underlying = closed_trades.groupby('Underlying').agg({
                'P_L_Amount': ['sum', 'mean'],
                'P_L_Percent': 'mean',
                'Symbol': 'count'
            })
            
            print("\n=== PERFORMANCE BY UNDERLYING ===")
            print(by_underlying)
            
            # Analyze by direction
            by_direction = closed_trades.groupby('Direction').agg({
                'P_L_Amount': ['sum', 'mean'],
                'P_L_Percent': 'mean',
                'Symbol': 'count'
            })
            
            print("\n=== PERFORMANCE BY DIRECTION ===")
            print(by_direction)
            
            # Analyze by exit reason
            by_exit = closed_trades.groupby('Exit_Reason').agg({
                'P_L_Amount': ['sum', 'mean'],
                'P_L_Percent': 'mean',
                'Symbol': 'count'
            })
            
            print("\n=== PERFORMANCE BY EXIT REASON ===")
            print(by_exit)
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_pl_amount': avg_pl_amount,
                'avg_pl_percent': avg_pl_percent,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            print(f"Error analyzing trading performance: {e}")
            return None

    def run_trading_cycle(self):
        try:
            self.reset_daily_trades()
            print("\n=== Running Options Trading Cycle ===")
            
            if self.should_close_expiring_positions():
                print("Closing positions expiring today...")
                self.close_expiring_positions()
            
            current_positions = self.api.list_positions()
            if len(current_positions) >= MAX_POSITIONS:
                print(f"Maximum positions ({MAX_POSITIONS}) reached. Not taking new trades.")
                return
            
            self.filtered_stocks = get_filtered_stocks(100)
            if not self.filtered_stocks:
                print("No stocks meet criteria. Waiting for next cycle....")
                return
            
            # Print performance analysis periodically
            if datetime.now().hour == 16 and datetime.now().minute < 15:  # Around market close
                self.analyze_trading_performance()
            
            for stock in self.filtered_stocks:
                symbol = stock['symbol']
                price = stock['price']
                direction = stock['direction']
                analysis = stock.get('analysis', {})
                trend_strength = stock.get('trend_strength', 0)
                momentum_score = stock.get('momentum_score', 0)
                risk_level = analysis.get('risk_level', stock.get('risk_level', 0))
                entry_signals = analysis.get('entry_signals', stock.get('entry_signals', []))
                
                if symbol in self._get_current_underlyings():
                    print(f"⚠️ Skipping {symbol} - open position exists")
                    continue
                if symbol in self.traded_today:
                    print(f"⚠️ Skipping {symbol} - already traded today")
                    continue
                
                # SIMPLIFIED ENTRY CRITERIA - Just check if direction is set
                if direction in ['CALL', 'PUT']:
                    print(f"✅ {symbol} meets entry criteria: Direction={direction}, Trend={trend_strength}, Momentum={momentum_score}, Risk={risk_level}")
                    trade_success = self.place_options_trade(symbol, price, direction)
                    if trade_success:
                        self.traded_today.add(symbol)
                        if len(self.api.list_positions()) >= MAX_POSITIONS:
                            print(f"Maximum positions ({MAX_POSITIONS}) reached. Stopping trading for now.")
                            break
                else:
                    print(f"❌ {symbol}: No trade - No clear direction signal")
                    
        except Exception as e:
            print(f"Trading cycle error: {e}")

def is_market_open(skip=False) -> bool:
    """Check if the market is currently open
    
    Args:
        skip (bool): If True, always return True (for testing purposes)
        
    Returns:
        bool: True if market is open or skip is True, False otherwise
    """
    if skip:
        return True
        
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    print(f"Current time: {now}")
    # Check if today is a weekday (Monday=0 to Friday=4)
    if now.weekday() >= 5:
        return False
    # Define market open and close times
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now <= close_time

def run_options_trading(test_mode=False):
    """Main trading loop with proper scheduling
    
    Args:
        test_mode (bool): If True, skip market open checks for testing purposes
    """
    executor = OptionsTradeExecutor()
    
    # Define the schedule
    def scheduled_trading_cycle():
        if is_market_open(test_mode):
            executor.run_trading_cycle()
        else:
            print("Market is closed. Skipping trading cycle.")
            
    def scheduled_monitoring():
        if is_market_open(test_mode):
            executor.monitor_positions()
        else:
            print("Market is closed. Skipping position monitoring.")
    
    # Run trading cycle immediately if market is open or in test mode
    print("Options trading bot started.")
    if is_market_open(test_mode):
        print("Market is open. Running initial trading cycle...")
        executor.run_trading_cycle()
        print("Initial monitoring of positions...")
        executor.monitor_positions()
    else:
        print("Market is closed. Waiting for market to open.")
    
    # Set up schedule
    schedule.every(15).minutes.do(scheduled_trading_cycle)  # Run scan every 15 minutes
    schedule.every(2).minutes.do(scheduled_monitoring)      # Monitor positions every 2 minutes
    
    print("Schedule set up. Bot will scan every 15 minutes and monitor positions every 2 minutes.")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)  # Check schedule every second

        except KeyboardInterrupt:
            print("\nStopping options trading...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Options Trading Bot')
    parser.add_argument('--test', action='store_true', help='Run in test mode (bypasses market hour checks)')
    
    args = parser.parse_args()
    
    # Run the trading bot with test mode if specified
    run_options_trading(test_mode=args.test)
