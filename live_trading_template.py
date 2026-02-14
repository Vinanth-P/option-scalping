"""
Live Trading Template for Gamma Scalping

This template shows how to adapt the backtesting system for live trading.
IMPORTANT: This is a template only. You must:
1. Connect to your broker's API for real data
2. Implement proper order execution
3. Add error handling and monitoring
4. Paper trade extensively before going live
"""

from gamma_scalping_system import GammaScalpingStrategy, BlackScholes
from datetime import datetime, timedelta
import pandas as pd
import time
import json

class LiveGammaScalper:
    """
    Live trading wrapper for gamma scalping strategy
    
    This class should be extended to work with your specific broker API
    """
    
    def __init__(self, strategy_params, broker_config):
        """
        Initialize live trading system
        
        Parameters:
        -----------
        strategy_params : dict
            Strategy configuration (delta_threshold, etc.)
        broker_config : dict
            Broker API credentials and settings
        """
        self.strategy = GammaScalpingStrategy(**strategy_params)
        self.broker_config = broker_config
        
        # Trading state
        self.is_running = False
        self.last_update = None
        self.iv_history = []
        
    def connect_broker_api(self):
        """
        Connect to broker API
        
        Replace with your broker's connection method:
        - Zerodha (Kite Connect)
        - Upstox
        - Angel Broking
        - IIFL
        - etc.
        """
        print("📡 Connecting to broker API...")
        
        # Example for Kite Connect (Zerodha):
        # from kiteconnect import KiteConnect
        # kite = KiteConnect(api_key=self.broker_config['api_key'])
        # 
        # # Generate session
        # data = kite.generate_session(
        #     self.broker_config['request_token'], 
        #     api_secret=self.broker_config['api_secret']
        # )
        # kite.set_access_token(data["access_token"])
        # self.kite = kite
        
        print("✅ Connected to broker")
        
    def get_nifty_spot_price(self):
        """
        Get current Nifty spot price
        
        Replace with actual API call
        """
        # Example for Kite:
        # quote = self.kite.quote("NSE:NIFTY 50")
        # return quote['NSE:NIFTY 50']['last_price']
        
        # Dummy for template
        return 21500.0
    
    def get_nifty_futures_price(self, expiry='current'):
        """
        Get Nifty futures price for delta hedging
        """
        # Example:
        # symbol = "NFO:NIFTY23DEC21500FUT"  # Construct based on expiry
        # quote = self.kite.quote(symbol)
        # return quote[symbol]['last_price']
        
        return 21510.0
    
    def get_option_chain(self):
        """
        Get current option chain with prices and Greeks
        
        Returns:
        --------
        dict : Option chain data with strikes, IVs, prices
        """
        # Example structure you need:
        # {
        #   21500: {
        #       'call_price': 150.5,
        #       'put_price': 145.3,
        #       'call_iv': 18.5,
        #       'put_iv': 19.2,
        #       'call_oi': 50000,
        #       'put_oi': 48000
        #   },
        #   21550: { ... },
        #   ...
        # }
        
        # For Kite, you'd fetch instruments and get quotes
        # instruments = self.kite.instruments("NFO")
        # Filter for NIFTY options with current weekly expiry
        # Get LTP, IV, OI for each strike
        
        # Dummy data for template
        return {
            21500: {
                'call_price': 150.0,
                'put_price': 145.0,
                'call_iv': 18.5,
                'put_iv': 19.0
            }
        }
    
    def get_atm_strike_iv(self):
        """
        Get implied volatility for ATM strike
        
        In practice, use average of call and put IV
        """
        spot = self.get_nifty_spot_price()
        option_chain = self.get_option_chain()
        
        # Find ATM strike
        atm_strike = self.strategy.get_atm_strike(spot)
        
        if atm_strike in option_chain:
            call_iv = option_chain[atm_strike]['call_iv']
            put_iv = option_chain[atm_strike]['put_iv']
            return (call_iv + put_iv) / 2
        
        return None
    
    def place_order(self, symbol, transaction_type, quantity, order_type='MARKET', price=None):
        """
        Place order through broker API
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., "NFO:NIFTY23DEC21500CE")
        transaction_type : str
            'BUY' or 'SELL'
        quantity : int
            Number of contracts
        order_type : str
            'MARKET' or 'LIMIT'
        price : float
            Limit price (for LIMIT orders)
        """
        print(f"📤 Placing order: {transaction_type} {quantity} {symbol} @ {order_type}")
        
        # Example for Kite:
        # order_id = self.kite.place_order(
        #     variety=self.kite.VARIETY_REGULAR,
        #     exchange=self.kite.EXCHANGE_NFO,
        #     tradingsymbol=symbol,
        #     transaction_type=transaction_type,
        #     quantity=quantity,
        #     product=self.kite.PRODUCT_NRML,
        #     order_type=order_type,
        #     price=price
        # )
        # 
        # return order_id
        
        # Dummy return
        return "ORDER_12345"
    
    def enter_straddle(self, spot, strike, call_price, put_price):
        """
        Enter straddle position by buying ATM call and put
        """
        # Construct option symbols based on your broker format
        # Zerodha format: NIFTY23DEC21500CE, NIFTY23DEC21500PE
        
        expiry_date = self.get_current_weekly_expiry()
        call_symbol = f"NIFTY{expiry_date}{strike}CE"
        put_symbol = f"NIFTY{expiry_date}{strike}PE"
        
        # Place orders
        call_order = self.place_order(call_symbol, 'BUY', 50, 'LIMIT', call_price * 1.01)
        put_order = self.place_order(put_symbol, 'BUY', 50, 'LIMIT', put_price * 1.01)
        
        print(f"✅ Straddle entered: {strike} @ Call={call_price}, Put={put_price}")
        
        return {
            'call_order': call_order,
            'put_order': put_order,
            'call_symbol': call_symbol,
            'put_symbol': put_symbol
        }
    
    def hedge_with_futures(self, quantity, transaction_type):
        """
        Hedge delta using Nifty futures
        
        Parameters:
        -----------
        quantity : int
            Number of futures lots (50 per lot for Nifty)
        transaction_type : str
            'BUY' or 'SELL'
        """
        futures_symbol = self.get_current_futures_symbol()
        
        order_id = self.place_order(futures_symbol, transaction_type, quantity, 'MARKET')
        
        print(f"⚖️ Delta hedged: {transaction_type} {quantity} futures")
        
        return order_id
    
    def exit_position(self, position_info):
        """
        Close entire position (options + futures)
        """
        # Sell options
        call_order = self.place_order(
            position_info['call_symbol'], 
            'SELL', 
            50, 
            'MARKET'
        )
        
        put_order = self.place_order(
            position_info['put_symbol'], 
            'SELL', 
            50, 
            'MARKET'
        )
        
        # Close futures hedge if any
        if position_info.get('futures_quantity', 0) != 0:
            hedge_type = 'SELL' if position_info['futures_quantity'] > 0 else 'BUY'
            hedge_qty = abs(position_info['futures_quantity'])
            self.place_order(self.get_current_futures_symbol(), hedge_type, hedge_qty, 'MARKET')
        
        print("🔚 Position closed")
    
    def get_current_weekly_expiry(self):
        """Get current weekly expiry date string"""
        # Nifty weekly expiry is Thursday
        # Return format like "23DEC" or "30DEC"
        today = datetime.now()
        days_until_thursday = (3 - today.weekday()) % 7
        expiry = today + timedelta(days=days_until_thursday)
        return expiry.strftime("%y%b").upper()
    
    def get_current_futures_symbol(self):
        """Get current month futures symbol"""
        expiry = self.get_current_weekly_expiry()
        return f"NFO:NIFTY{expiry}FUT"
    
    def calculate_position_pnl(self, position_info):
        """Calculate current position P&L"""
        # Get current option prices
        option_chain = self.get_option_chain()
        strike = position_info['strike']
        
        if strike in option_chain:
            current_call = option_chain[strike]['call_price']
            current_put = option_chain[strike]['put_price']
            
            call_pnl = (current_call - position_info['call_entry']) * 50
            put_pnl = (current_put - position_info['put_entry']) * 50
            
            # Futures P&L
            futures_pnl = 0
            if position_info.get('futures_quantity', 0) != 0:
                current_futures = self.get_nifty_futures_price()
                futures_pnl = position_info['futures_quantity'] * (
                    current_futures - position_info['futures_entry']
                ) * 50
            
            total_pnl = call_pnl + put_pnl + futures_pnl
            
            return {
                'call_pnl': call_pnl,
                'put_pnl': put_pnl,
                'futures_pnl': futures_pnl,
                'total_pnl': total_pnl
            }
        
        return None
    
    def monitor_and_trade(self, update_interval=60):
        """
        Main trading loop
        
        Parameters:
        -----------
        update_interval : int
            Seconds between updates (60 = check every minute)
        """
        print("🚀 Starting live gamma scalping...")
        print(f"Update interval: {update_interval} seconds")
        
        self.is_running = True
        position_info = None
        
        try:
            while self.is_running:
                # Get current market data
                spot = self.get_nifty_spot_price()
                current_iv = self.get_atm_strike_iv()
                
                if current_iv is None:
                    print("⚠️ Unable to get IV data, retrying...")
                    time.sleep(update_interval)
                    continue
                
                # Update IV history
                self.iv_history.append(current_iv)
                if len(self.iv_history) > 100:
                    self.iv_history.pop(0)
                
                iv_series = pd.Series(self.iv_history)
                
                # Check if we have a position
                if position_info is None:
                    # Check entry conditions
                    if self.strategy.should_enter(current_iv, iv_series):
                        print(f"\n💡 Entry signal detected!")
                        print(f"   Spot: {spot:.2f}")
                        print(f"   IV: {current_iv:.2f}%")
                        
                        # Get option prices
                        strike = self.strategy.get_atm_strike(spot)
                        option_chain = self.get_option_chain()
                        
                        if strike in option_chain:
                            call_price = option_chain[strike]['call_price']
                            put_price = option_chain[strike]['put_price']
                            
                            # Enter position
                            order_info = self.enter_straddle(spot, strike, call_price, put_price)
                            
                            position_info = {
                                'entry_time': datetime.now(),
                                'strike': strike,
                                'spot_entry': spot,
                                'call_entry': call_price,
                                'put_entry': put_price,
                                'call_symbol': order_info['call_symbol'],
                                'put_symbol': order_info['put_symbol'],
                                'futures_quantity': 0,
                                'futures_entry': 0,
                                'entry_iv': current_iv
                            }
                else:
                    # Monitor existing position
                    pnl_data = self.calculate_position_pnl(position_info)
                    
                    if pnl_data:
                        print(f"\n📊 Position Update:")
                        print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
                        print(f"   Spot: {spot:.2f}")
                        print(f"   Total P&L: ₹{pnl_data['total_pnl']:.2f}")
                        
                        # Calculate Greeks and check for hedging
                        T = max(0, (position_info['entry_time'] + timedelta(days=7) - datetime.now()).days / 365)
                        greeks = self.strategy.calculate_position_greeks(
                            spot, position_info['strike'], T, current_iv
                        )
                        
                        print(f"   Portfolio Delta: {greeks['delta']:.3f}")
                        print(f"   Gamma: {greeks['gamma']:.4f}")
                        
                        # Check for rehedge
                        if self.strategy.should_rehedge(greeks['delta']):
                            print(f"\n⚖️ Rehedge triggered! Delta: {greeks['delta']:.3f}")
                            
                            # Calculate required hedge
                            required_hedge = -greeks['delta']
                            current_hedge = position_info.get('futures_quantity', 0)
                            delta_hedge = required_hedge - current_hedge
                            
                            # Place hedge order
                            if delta_hedge > 0:
                                self.hedge_with_futures(abs(int(delta_hedge * 50)), 'BUY')
                            else:
                                self.hedge_with_futures(abs(int(delta_hedge * 50)), 'SELL')
                            
                            position_info['futures_quantity'] = required_hedge
                            position_info['futures_entry'] = self.get_nifty_futures_price()
                        
                        # Check exit conditions
                        premium_paid = (position_info['call_entry'] + position_info['put_entry']) * 50
                        pnl_pct = pnl_data['total_pnl'] / premium_paid
                        
                        if self.strategy.should_exit(current_iv, iv_series, pnl_pct):
                            reason = 'PROFIT' if pnl_pct > 0 else 'LOSS'
                            print(f"\n🚪 Exit signal: {reason}")
                            print(f"   P&L: ₹{pnl_data['total_pnl']:.2f} ({pnl_pct*100:.1f}%)")
                            
                            self.exit_position(position_info)
                            position_info = None
                
                # Wait for next update
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopping trading loop...")
            if position_info:
                print("⚠️ Active position detected! Please close manually.")
        
        self.is_running = False


def main():
    """
    Main entry point for live trading
    """
    
    # Strategy configuration
    strategy_params = {
        'delta_threshold': 0.15,
        'iv_entry_percentile': 30,
        'iv_exit_percentile': 70,
        'profit_target': 0.50,
        'max_loss': -0.30,
        'time_to_expiry': 7,
        'risk_free_rate': 0.06
    }
    
    # Broker configuration (REPLACE WITH YOUR CREDENTIALS)
    broker_config = {
        'api_key': 'YOUR_API_KEY',
        'api_secret': 'YOUR_API_SECRET',
        'request_token': 'YOUR_REQUEST_TOKEN'
    }
    
    # Initialize live trader
    trader = LiveGammaScalper(strategy_params, broker_config)
    
    # Connect to broker
    trader.connect_broker_api()
    
    # Start monitoring and trading
    trader.monitor_and_trade(update_interval=60)  # Check every 60 seconds


if __name__ == "__main__":
    print("""
    ⚠️  WARNING: LIVE TRADING TEMPLATE ⚠️
    
    This is a TEMPLATE only. Before using:
    
    1. Implement actual broker API integration
    2. Add proper error handling
    3. Add logging and monitoring
    4. Test extensively with paper trading
    5. Start with small position sizes
    6. Monitor continuously
    
    Press Ctrl+C to stop at any time.
    
    """)
    
    response = input("Have you read and understood the warnings? (yes/no): ")
    
    if response.lower() == 'yes':
        main()
    else:
        print("Please review the code and warnings before proceeding.")
