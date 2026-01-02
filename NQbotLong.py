import yfinance as yf, pandas as pd, numpy as np, datetime as dt, joblib, os, csv, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Data Download and Feature Engineering (from your Colab notebook) ---
# Download data for training the model
df = yf.download('NQ=F', start=dt.date.today()-dt.timedelta(days=30),
                 interval='15m', prepost=False, progress=False)
df = df[['Open','High','Low','Close','Volume']].rename(columns=str.lower)

# Calculate features (returns, RSI, ATR, EMAs)
df['ret'] = df['close'].pct_change()
delta = df['close'].diff()
gain  = delta.clip(lower=0)
loss  = -delta.clip(upper=0)
rs    = gain.rolling(14).mean() / loss.rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + rs))
tr = np.maximum(df['high'] - df['low'],
                np.maximum(abs(df['high'] - df['close'].shift(1)),
                           abs(df['low']  - df['close'].shift(1))))
df['atr'] = tr.rolling(14).mean()
df['ema8']  = df['close'].ewm(span=8,  adjust=False).mean()
df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
df = df.dropna()

# Define target variable
df['target'] = (df['close'].squeeze().shift(-1) - df['close'].squeeze()) >= 0.25 * df['atr'].squeeze()
df['target'] = df['target'].astype(int)
df = df.dropna()

# --- Model Training ---
X = df[['ret','rsi','atr','ema8','ema21']]
y = df['target']
split = int(len(df)*0.8)
clf = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=42)
clf.fit(X[:split], y[:split])
pred = clf.predict(X[split:])
print('Training Accuracy:', accuracy_score(y[split:], pred))

# --- Model Saving (local path) ---
LOCAL_MODEL_DIR = './nqbot'
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, 'nq_rf_v1.joblib')
joblib.dump(clf, MODEL_PATH)
print(f'Model saved to {MODEL_PATH}')

# --- Live Trading Bot Logic ---
# Define local log path
LOCAL_LOG_PATH = os.path.join(LOCAL_MODEL_DIR, 'reg_live_trades.csv')
if not os.path.exists(LOCAL_LOG_PATH):
    with open(LOCAL_LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['timestamp', 'event', 'pred_proba', 'price', 'TP', 'SL', 'PnL_points'])

# Helper function to round to nearest 0.25
def round_to_quarter(price):
    return round(price * 4) / 4

# Define constants (matching backtester settings)
PROBABILITY_THRESHOLD = 0.6
SLIPPAGE_AMOUNT = 2.0
MIN_PROFIT_TARGET_POINTS = 5.0
TRAILING_STOP_ATR_MULTIPLIER = 1.0
EXIT_PROBABILITY_THRESHOLD = 0.45

# --- State Variables for Live Trading ---
is_trade_open = False
open_trade = {
    'entry_time': None,
    'entry_price': None,
    'entry_proba': None,
    'initial_tp': None,
    'initial_sl': None, # This is the original calculated SL, for reference
    'active_sl': None,  # This will be updated for trailing stop
    'max_high_since_entry': None,
}

# sigma for printing (using the validation set from training)
sigma = np.std(y[split:])

print(f"Trading bot started with PROBABILITY_THRESHOLD={PROBABILITY_THRESHOLD}, EXIT_PROBABILITY_THRESHOLD={EXIT_PROBABILITY_THRESHOLD}, TRAILING_STOP_ATR_MULTIPLIER={TRAILING_STOP_ATR_MULTIPLIER}")

while True:
    current_time_str = dt.datetime.now().isoformat()
    try:
        # 1-min bar
        bar = yf.download('NQ=F', period='1d', interval='1m', prepost=False, progress=False)
        bar = bar[['Open','High','Low','Close','Volume']].rename(columns=str.lower)
        bar.columns = bar.columns.get_level_values(0)

        # Ensure we have enough data (at least 15 bars for 14-period indicators)
        if len(bar) < 15:
            print(f"[{current_time_str}] Not enough historical 1m data for indicators. Waiting...")
            time.sleep(60)
            continue

        # feats
        bar['ret'] = bar['close'].pct_change()
        delta = bar['close'].diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean(); bar['rsi'] = 100 - (100 / (1 + rs))
        tr = np.maximum(bar['high'] - bar['low'], np.maximum(abs(bar['high'] - bar['close'].shift(1)), abs(bar['low'] - bar['close'].shift(1))))
        bar['atr'] = tr.rolling(14).mean(); bar['ema8'] = bar['close'].ewm(span=8, adjust=False).mean(); bar['ema21'] = bar['close'].ewm(span=21, adjust=False).mean()

        # Get latest bar's data and features
        current_bar_data = bar.iloc[-1]
        feat = current_bar_data[['ret','rsi','atr','ema8','ema21']].to_frame().T

        # Skip if latest features contain NaN
        if feat.isnull().values.any():
            print(f"[{current_time_str}] Latest bar features contain NaN. Waiting...")
            time.sleep(60)
            continue

        pred_proba = clf.predict_proba(feat.values)[0][1]
        current_close = current_bar_data['close'].item()
        current_high = current_bar_data['high'].item()
        current_low = current_bar_data['low'].item()
        current_atr = current_bar_data['atr'].item()

        print(f"[{current_time_str}] Current Close: {current_close:.2f} | raw pred: {pred_proba:.6f} | Sigma: {sigma:.2f} | 0.25*Sigma: {0.25*sigma:.2f}")

        if not is_trade_open: # Looking for an entry
            if pred_proba >= PROBABILITY_THRESHOLD:
                # Calculate initial TP/SL with slippage and quarter rounding
                risk_reward_unit = max(current_atr, MIN_PROFIT_TARGET_POINTS / 1.5)

                entry_slipped = current_close + SLIPPAGE_AMOUNT # Buy at slightly higher
                tp_slipped = current_close + 1.5 * risk_reward_unit - SLIPPAGE_AMOUNT
                sl_slipped = current_close - 1.0 * risk_reward_unit - SLIPPAGE_AMOUNT

                entry_final = round_to_quarter(entry_slipped)
                tp_final = round_to_quarter(tp_slipped)
                sl_final = round_to_quarter(sl_slipped)

                is_trade_open = True
                open_trade['entry_time'] = current_time_str
                open_trade['entry_price'] = entry_final
                open_trade['entry_proba'] = pred_proba
                open_trade['initial_tp'] = tp_final
                open_trade['initial_sl'] = sl_final
                open_trade['active_sl'] = sl_final
                open_trade['max_high_since_entry'] = current_high # Initialize with current high

                log_row = [
                    current_time_str, 'ENTRY', f'{pred_proba:.4f}', f'{entry_final:.2f}',
                    f'{tp_final:.2f}', f'{sl_final:.2f}', '0.00' # PnL is 0 at entry
                ]
                with open(LOCAL_LOG_PATH, 'a', newline='') as f:
                    csv.writer(f).writerow(log_row)
                print(f"[{current_time_str}] -> ENTRY logged: Price={entry_final:.2f}, TP={tp_final:.2f}, SL={sl_final:.2f}")
            else:
                print(f"[{current_time_str}] Skip - Pred too small for entry.")

        else: # Trade is open, monitor for exit
            trade_entry_price = open_trade['entry_price']
            trade_initial_tp = open_trade['initial_tp']
            trade_active_sl = open_trade['active_sl']
            trade_max_high_since_entry = open_trade['max_high_since_entry']

            # Update max high for trailing stop
            trade_max_high_since_entry = max(trade_max_high_since_entry, current_high)

            # Calculate new trailing stop
            new_trailing_sl = round_to_quarter(trade_max_high_since_entry - (TRAILING_STOP_ATR_MULTIPLIER * current_atr))

            # Update active stop loss (it can only move up, or stay same if new_trailing_sl is lower than current active_sl)
            trade_active_sl = max(trade_active_sl, new_trailing_sl)

            # Store updated active_sl and max_high_since_entry back into open_trade state
            open_trade['max_high_since_entry'] = trade_max_high_since_entry
            open_trade['active_sl'] = trade_active_sl

            outcome = None
            pnl_points = 0.0
            exit_price = current_close # Default exit price if no specific level hit

            # Check exit conditions (order matters: SL, TP, then Prob Exit)
            if current_low <= trade_active_sl: # Trailing Stop or Initial SL hit
                outcome = 'LOSS'
                pnl_points = trade_active_sl - trade_entry_price
                exit_price = trade_active_sl
            elif current_high >= trade_initial_tp: # Take Profit hit
                outcome = 'WIN'
                pnl_points = trade_initial_tp - trade_entry_price
                exit_price = trade_initial_tp
            elif pred_proba < EXIT_PROBABILITY_THRESHOLD: # Probability exit
                outcome = 'PROB_EXIT'
                pnl_points = current_close - trade_entry_price
                exit_price = current_close

            if outcome:
                log_row = [
                    current_time_str, outcome, f'{pred_proba:.4f}', f'{exit_price:.2f}',
                    f'{open_trade["initial_tp"]:.2f}', f'{open_trade["initial_sl"]:.2f}', f'{pnl_points:.2f}'
                ]
                with open(LOCAL_LOG_PATH, 'a', newline='') as f:
                    csv.writer(f).writerow(log_row)
                print(f"[{current_time_str}] -> {outcome} logged: Exit Price={exit_price:.2f}, PnL={pnl_points:.2f}")
                is_trade_open = False
                open_trade = { # Reset trade state
                    'entry_time': None, 'entry_price': None, 'entry_proba': None,
                    'initial_tp': None, 'initial_sl': None, 'active_sl': None,
                    'max_high_since_entry': None,
                }
            else:
                print(f"[{current_time_str}] Trade Open. Current SL: {trade_active_sl:.2f}, Current TP: {trade_initial_tp:.2f}, Max High: {trade_max_high_since_entry:.2f}, Pred Proba: {pred_proba:.4f}")

    except Exception as e:
        print(f"[{current_time_str}] An error occurred: {e}")
        # Optionally, log the full traceback for debugging
        # import traceback
        # traceback.print_exc()

    time.sleep(60) # Poll every 1 minute
