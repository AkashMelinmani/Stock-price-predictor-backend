import yfinance as yf
import numpy as np
import pmdarima as pm


def get_data(ticker: str, period: str = "180d", interval: str = "1d"):
    """Download recent historical data."""
    data = yf.download(ticker, period=period, interval=interval)
    return data["Close"].dropna()


def train_model(series):
    """Fit an auto_arima model on log prices."""
    ts_log = np.log(series)

    model = pm.auto_arima(
        ts_log,
        seasonal=False,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    return model


def predict_next_close(model):
    """Predict next closing price from the ARIMA model."""
    forecast_log = model.predict(n_periods=1)

    # forecast_log can be ndarray or Series – handle both safely
    try:
        predicted_log = forecast_log.iloc[0]
    except AttributeError:
        predicted_log = forecast_log[0]

    predicted_price = float(np.exp(predicted_log))
    return predicted_price


def get_live_price(ticker: str):
    """Get current live/near‑real‑time price using yfinance."""
    info = yf.Ticker(ticker).fast_info
    current_price = info.last_price
    prev_close = info.previous_close
    return current_price, prev_close


def main():
    ticker = "AAPL"  # change to any ticker, e.g. "RELIANCE.NS"

    # 1. Download data
    close_series = get_data(ticker)

    # 2. Train ARIMA
    model = train_model(close_series)

    # 3. Predict next close
    predicted_price = predict_next_close(model)

    # 4. Get live price
    current_price, prev_close = get_live_price(ticker)

    # 5. Print results
    print(f"\nTicker: {ticker}")
    print(f"Current live price:        {current_price:.2f}")
    print(f"Previous close (Yahoo):    {prev_close:.2f}")
    print(f"Predicted next close ARIMA:{predicted_price:.2f}\n")


if __name__ == "__main__":
    main()


