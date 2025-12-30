import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm

# 1. Download stock data
ticker = input("Enter the stock symbol")   # change to any symbol
start_date = "2018-01-01"
end_date = "2025-01-01"

data = yf.download(ticker, start=start_date, end=end_date)
ts = data["Close"].dropna()

# 2. Take log to stabilize variance (optional but common)
ts_log = np.log(ts)

# 3. Let auto_arima find best (p,d,q)
model = pm.auto_arima(
    ts_log,
    seasonal=False,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)

print(model.summary())

# 4. Forecast future prices
n_steps = 30  # days to predict
forecast_log = model.predict(n_periods=n_steps)

# convert back from log scale
forecast = np.exp(forecast_log)

# build forecast index (next business days)
forecast_index = pd.date_range(
    start=ts.index[-1] + pd.Timedelta(days=1),
    periods=n_steps,
    freq="B"
)
forecast_series = pd.Series(forecast, index=forecast_index)

# ---- NEW: print predicted prices in terminal ----
print("\nPredicted prices for the next", n_steps, "business days:")
for date, price in forecast_series.items():
    print(f"{date.date()} -> {price:.2f}")

# Also print only the next day's predicted price
next_day = forecast_series.index[0]
next_price = forecast_series.iloc[0]
print(f"\nPredicted next closing price on {next_day.date()}: {next_price:.2f}")

# 5. Plot
plt.figure(figsize=(10, 5))
plt.plot(ts[-200:], label="Historical Close")
plt.plot(forecast_series, label="auto_arima forecast", color="red")
plt.title(f"{ticker} stock price forecast (auto_arima)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
