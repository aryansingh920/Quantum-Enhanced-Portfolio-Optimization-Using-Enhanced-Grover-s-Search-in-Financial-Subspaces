"""
Fixed version of yf_fetcher.py
- Fetches AAPL data
- Computes indicators
- Normalizes features
- Saves CSV + plots all columns as subgraphs
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import date

# ────────────────────────────────────────────────────────────────────────────────
TICKER = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
START = "2000-01-01"
END = str(date.today())

# ─── 1. Fetch Data ──────────────────────────────────────────────────────────────
print(f"Fetching {TICKER} from {START} to {END} …")
data = yf.download(TICKER, start=START, end=END,
                   interval="1d", auto_adjust=True)

# Fix potential MultiIndex column names
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0].lower() for col in data.columns]
else:
    data.columns = [col.lower() for col in data.columns]

data.dropna(inplace=True)

# ─── 2. Indicators ──────────────────────────────────────────────────────────────
print("Calculating technical indicators …")
data["ma5"] = data["close"].rolling(window=5).mean()
data["ma20"] = data["close"].rolling(window=20).mean()

delta = data["close"].diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
ema_up = up.ewm(span=14, adjust=False).mean()
ema_down = down.ewm(span=14, adjust=False).mean()
rs = ema_up / ema_down
data["rsi"] = 100 - (100 / (1 + rs))

data["returns"] = data["close"].pct_change()
data["volatility"] = data["returns"].rolling(window=10).std()

# ─── 3. Normalize ───────────────────────────────────────────────────────────────
features = ["open", "high", "low", "close",
            "volume", "ma5", "ma20", "rsi", "volatility"]
scaler = StandardScaler()
scaled = pd.DataFrame(scaler.fit_transform(data[features]),
                      columns=[f"{col}_scaled" for col in features],
                      index=data.index)

# ─── 4. Label regimes ───────────────────────────────────────────────────────────
data["regime"] = (data["volatility"] > data["volatility"].median()).astype(int)

# ─── 5. Merge + Drop NaNs ───────────────────────────────────────────────────────
final_df = pd.concat([data, scaled], axis=1)
final_df.dropna(inplace=True)

# ─── 6. Save CSV ────────────────────────────────────────────────────────────────
output_dir = f"ticker/{TICKER.upper()}"
os.makedirs(output_dir, exist_ok=True)
csv_path = f"{output_dir}/{TICKER.upper()}_data.csv"
final_df.to_csv(csv_path)
print(f"Saved final data to: {csv_path}")

# ─── 7. Plot all columns ────────────────────────────────────────────────────────
graphs_dir = f"{output_dir}/graphs"
os.makedirs(graphs_dir, exist_ok=True)

print(f"Generating plots in: {graphs_dir}")
plt.ioff()
for col in final_df.columns:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(final_df.index, final_df[col])
    ax.set_title(f"{TICKER.upper()} – {col}")
    ax.set_xlabel("Date")
    ax.set_ylabel(col)
    fig.tight_layout()
    fig.savefig(f"{graphs_dir}/{col}.png")
    plt.close(fig)

print("✅ All graphs and CSV saved.")
