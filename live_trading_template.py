"""
Live Trading Template — Gamma Scalping
Connect to your broker API, implement order execution, and paper trade before going live.
"""

from gamma_scalping_system import GammaScalpingStrategy, BlackScholes
from datetime import datetime, timedelta
import pandas as pd
import time
import json


class LiveGammaScalper:
    """Live trading wrapper for gamma scalping strategy."""

    def __init__(self, strategy_params, broker_config):
        self.strategy = GammaScalpingStrategy(**strategy_params)
        self.broker_config = broker_config
        self.is_running = False
        self.last_update = None
        self.iv_history = []

    def connect_broker_api(self):
        """Connect to broker API (Zerodha/Upstox/Angel/IIFL). Implement your connection here."""
        print("📡 Connecting to broker API...")
        # Example: kite = KiteConnect(api_key=self.broker_config['api_key'])
        print("✅ Connected to broker")

    def get_nifty_spot_price(self):
        """Get current Nifty spot price. Replace with actual API call."""
        return 21500.0

    def get_nifty_futures_price(self, expiry='current'):
        """Get Nifty futures price for delta hedging."""
        return 21510.0

    def get_option_chain(self):
        """Get current option chain with prices and Greeks."""
        return {
            21500: {
                'call_price': 150.0, 'put_price': 145.0,
                'call_iv': 18.5, 'put_iv': 19.0
            }
        }

    def get_atm_strike_iv(self):
        """Get ATM implied volatility (average of call and put IV)."""
        spot = self.get_nifty_spot_price()
        option_chain = self.get_option_chain()
        atm_strike = self.strategy.get_atm_strike(spot)
        if atm_strike in option_chain:
            return (option_chain[atm_strike]['call_iv'] + option_chain[atm_strike]['put_iv']) / 2
        return None

    def place_order(self, symbol, transaction_type, quantity, order_type='MARKET', price=None):
        """Place order through broker API. Implement your broker's order method here."""
        print(f"📤 Placing order: {transaction_type} {quantity} {symbol} @ {order_type}")
        return "ORDER_12345"

    def enter_straddle(self, spot, strike, call_price, put_price):
        """Enter straddle by buying ATM call and put."""
        expiry_date = self.get_current_weekly_expiry()
        call_symbol = f"NIFTY{expiry_date}{strike}CE"
        put_symbol = f"NIFTY{expiry_date}{strike}PE"

        call_order = self.place_order(call_symbol, 'BUY', 50, 'LIMIT', call_price * 1.01)
        put_order = self.place_order(put_symbol, 'BUY', 50, 'LIMIT', put_price * 1.01)
        print(f"✅ Straddle entered: {strike} @ Call={call_price}, Put={put_price}")

        return {
            'call_order': call_order, 'put_order': put_order,
            'call_symbol': call_symbol, 'put_symbol': put_symbol
        }

    def hedge_with_futures(self, quantity, transaction_type):
        """Hedge delta using Nifty futures."""
        futures_symbol = self.get_current_futures_symbol()
        order_id = self.place_order(futures_symbol, transaction_type, quantity, 'MARKET')
        print(f"⚖️ Delta hedged: {transaction_type} {quantity} futures")
        return order_id

    def exit_position(self, position_info):
        """Close entire position (options + futures)."""
        self.place_order(position_info['call_symbol'], 'SELL', 50, 'MARKET')
        self.place_order(position_info['put_symbol'], 'SELL', 50, 'MARKET')

        if position_info.get('futures_quantity', 0) != 0:
            hedge_type = 'SELL' if position_info['futures_quantity'] > 0 else 'BUY'
            self.place_order(self.get_current_futures_symbol(), hedge_type,
                           abs(position_info['futures_quantity']), 'MARKET')
        print("🔚 Position closed")

    def get_current_weekly_expiry(self):
        """Get current weekly expiry date string (Nifty weekly = Thursday)."""
        today = datetime.now()
        days_until_thursday = (3 - today.weekday()) % 7
        expiry = today + timedelta(days=days_until_thursday)
        return expiry.strftime("%y%b").upper()

    def get_current_futures_symbol(self):
        return f"NFO:NIFTY{self.get_current_weekly_expiry()}FUT"

    def calculate_position_pnl(self, position_info):
        """Calculate current position P&L."""
        option_chain = self.get_option_chain()
        strike = position_info['strike']

        if strike in option_chain:
            call_pnl = (option_chain[strike]['call_price'] - position_info['call_entry']) * 50
            put_pnl = (option_chain[strike]['put_price'] - position_info['put_entry']) * 50

            futures_pnl = 0
            if position_info.get('futures_quantity', 0) != 0:
                futures_pnl = position_info['futures_quantity'] * (
                    self.get_nifty_futures_price() - position_info['futures_entry']
                ) * 50

            return {'call_pnl': call_pnl, 'put_pnl': put_pnl,
                    'futures_pnl': futures_pnl, 'total_pnl': call_pnl + put_pnl + futures_pnl}
        return None

    def monitor_and_trade(self, update_interval=60):
        """Main trading loop — monitors market and executes strategy."""
        print(f"🚀 Starting live gamma scalping (interval: {update_interval}s)...")
        self.is_running = True
        position_info = None

        try:
            while self.is_running:
                spot = self.get_nifty_spot_price()
                current_iv = self.get_atm_strike_iv()

                if current_iv is None:
                    print("⚠️ Unable to get IV data, retrying...")
                    time.sleep(update_interval)
                    continue

                self.iv_history.append(current_iv)
                if len(self.iv_history) > 100:
                    self.iv_history.pop(0)
                iv_series = pd.Series(self.iv_history)

                if position_info is None:
                    if self.strategy.should_enter(current_iv, iv_series):
                        print(f"\n💡 Entry signal! Spot: {spot:.2f}, IV: {current_iv:.2f}%")
                        strike = self.strategy.get_atm_strike(spot)
                        option_chain = self.get_option_chain()

                        if strike in option_chain:
                            call_price = option_chain[strike]['call_price']
                            put_price = option_chain[strike]['put_price']
                            order_info = self.enter_straddle(spot, strike, call_price, put_price)

                            position_info = {
                                'entry_time': datetime.now(), 'strike': strike,
                                'spot_entry': spot, 'call_entry': call_price,
                                'put_entry': put_price,
                                'call_symbol': order_info['call_symbol'],
                                'put_symbol': order_info['put_symbol'],
                                'futures_quantity': 0, 'futures_entry': 0,
                                'entry_iv': current_iv
                            }
                else:
                    pnl_data = self.calculate_position_pnl(position_info)
                    if pnl_data:
                        print(f"\n📊 Spot: {spot:.2f} | P&L: ₹{pnl_data['total_pnl']:.2f}")

                        T = max(0, (position_info['entry_time'] + timedelta(days=7) - datetime.now()).days / 365)
                        greeks = self.strategy.calculate_position_greeks(
                            spot, position_info['strike'], T, current_iv
                        )
                        print(f"   Delta: {greeks['delta']:.3f} | Gamma: {greeks['gamma']:.4f}")

                        # Check for rehedge
                        if self.strategy.should_rehedge(greeks['delta'], spot, timestamp=datetime.now()):
                            print(f"\n⚖️ Rehedge triggered! Delta: {greeks['delta']:.3f}")
                            required_hedge = -greeks['delta']
                            delta_hedge = required_hedge - position_info.get('futures_quantity', 0)

                            if delta_hedge > 0:
                                self.hedge_with_futures(abs(int(delta_hedge * 50)), 'BUY')
                            else:
                                self.hedge_with_futures(abs(int(delta_hedge * 50)), 'SELL')

                            position_info['futures_quantity'] = required_hedge
                            position_info['futures_entry'] = self.get_nifty_futures_price()

                        # Check exit
                        premium_paid = (position_info['call_entry'] + position_info['put_entry']) * 50
                        pnl_pct = pnl_data['total_pnl'] / premium_paid

                        if self.strategy.should_exit(current_iv, iv_series, pnl_pct):
                            reason = 'PROFIT' if pnl_pct > 0 else 'LOSS'
                            print(f"\n🚪 Exit: {reason} | P&L: ₹{pnl_data['total_pnl']:.2f} ({pnl_pct*100:.1f}%)")
                            self.exit_position(position_info)
                            position_info = None

                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\n⏹️ Stopping...")
            if position_info:
                print("⚠️ Active position detected! Close manually.")
        self.is_running = False


def main():
    strategy_params = {
        'delta_hedge_band': 0.25, 'spot_move_threshold': 0.0025,
        'rv_iv_edge_min': 0.05, 'rv_iv_edge_max': 0.10,
        'iv_entry_percentile': 65, 'rv_window_days': 5,
        'noon_rv_threshold': 0.45, 'afternoon_rv_threshold': 0.70,
        'iv_drop_exit': 0.08, 'delta_pin_threshold': 0.50,
        'delta_pin_duration_minutes': 30, 'profit_target': 0.50, 'max_loss': -0.30,
        'time_to_expiry': 7, 'risk_free_rate': 0.06,
        'first_15m_move_threshold': 0.0020, 'round_trip_cost': 285.0,
        'economic_multiplier': 4.0, 'hedge_cooldown_minutes': 7, 'max_daily_hedges': 30,
    }

    broker_config = {
        'api_key': 'YOUR_API_KEY',
        'api_secret': 'YOUR_API_SECRET',
        'request_token': 'YOUR_REQUEST_TOKEN'
    }

    trader = LiveGammaScalper(strategy_params, broker_config)
    trader.connect_broker_api()
    trader.monitor_and_trade(update_interval=60)


if __name__ == "__main__":
    print("""
    ⚠️  LIVE TRADING TEMPLATE — Read before using:
    1. Implement broker API integration
    2. Add error handling and logging
    3. Paper trade extensively first
    4. Start with small positions
    Press Ctrl+C to stop.
    """)
    if input("Understood? (yes/no): ").lower() == 'yes':
        main()
    else:
        print("Please review before proceeding.")
