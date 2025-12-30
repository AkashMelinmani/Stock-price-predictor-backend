from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import pmdarima as pm

app = FastAPI(title="Stock ARIMA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    ticker: str


def get_close_series(ticker: str, period="180d", interval="1d"):
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False
    )

    if data.empty or "Close" not in data:
        raise ValueError("No price data returned from Yahoo Finance")

    series = data["Close"].dropna()

    if len(series) < 30:
        raise ValueError("Not enough historical data")

    return series


def train_model(series):
    ts_log = np.log(series.replace(0, np.nan).dropna())

    model = pm.auto_arima(
        ts_log,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3,
        max_q=3,
    )
    return model


def predict_next_close(model):
    forecast_log = model.predict(n_periods=1)

    if isinstance(forecast_log, (list, np.ndarray)):
        predicted_log = forecast_log[0]
    else:
        predicted_log = forecast_log.iloc[0]

    return float(np.exp(predicted_log))


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        ticker = req.ticker.upper().strip()

        # Fetch historical data
        series = get_close_series(ticker)

        # Train model
        model = train_model(series)
        predicted_price = predict_next_close(model)

        # Last close
        last_close = float(series.iloc[-1])

        # ✅ FIX: define info properly
        info = yf.Ticker(ticker).fast_info
        currency = info.get("currency", "USD")

        # fallback for Indian stocks
        if ticker.endswith(".NS") or ticker.endswith(".BO"):
            currency = "INR"

        return {
            "ticker": ticker,
            "last_close": round(last_close, 2),
            "predicted_next_close": round(predicted_price, 2),
            "currency": currency,
            "horizon": "Next trading day",
            "model": "ARIMA (log-transformed)",
            "status": "success"
        }
        from fastapi import Query

        @app.get("/history")
        def get_stock_history(
        ticker: str = Query(...),
        period: str = "1mo",
        interval: str = "1d"
    ):
          data = yf.download(ticker, period=period, interval=interval)

          data = data.reset_index()

          result = []
        for _, row in data.iterrows():
             result.append({
             "date": str(row["Date"].date()),
            "close": float(row["Close"])
        })

        return result


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
