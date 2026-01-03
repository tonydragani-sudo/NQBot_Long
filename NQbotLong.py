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
#Proprietary Logic - model weights and features are removed for IP protection

# Helper function to round to nearest 0.25
def round_to_quarter(price):
    return round(price * 4) / 4

# Define constants (matching backtester settings)
#Proprietary Logic - model weights and features are removed for IP protection

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
#Proprietary Logic - model weights and features are removed for IP protection

    time.sleep(60) # Poll every 1 minute


