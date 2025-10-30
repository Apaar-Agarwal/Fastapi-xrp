# app/main.py
# 1) Imports
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib, os
from typing import Optional

# 2) Constants / config
APP_TITLE = "XRP Next-Day High Predictor"
FEATURE_COLS = [
    "open","low","close","volume","marketcap",
    "price_change","volatility_abs","ret_1d","ret_7d",
    "ma_7","ma_30","vol_ma_7","vol_ma_30"
]
MODEL_PATH = os.getenv("MODEL_PATH", "models/xrp_xgb_model.joblib")
ARIMA_PATH = os.getenv("ARIMA_PATH", "models/arima_xrp.pkl")  # optional

# 3) Load models
model = joblib.load(MODEL_PATH)
arima_model = None
if os.path.exists(ARIMA_PATH):
    try:
        import pickle
        with open(ARIMA_PATH, "rb") as f:
            arima_model = pickle.load(f)
    except Exception:
        arima_model = None

# 4) FastAPI app
app = FastAPI(title=APP_TITLE, version=os.getenv("MODEL_VERSION", "v1.0"))

# 5) Request schema
class XRPFeatures(BaseModel):
    open: float
    low: float
    close: float
    volume: float
    marketcap: float
    price_change: float
    volatility_abs: float
    ret_1d: float
    ret_7d: float
    ma_7: float
    ma_30: float
    vol_ma_7: float
    vol_ma_30: float
    use_arima: Optional[bool] = Field(default=False)

# 6) Routes  ←←← YOUR BLOCK GOES HERE (with the missing decorator added)
@app.get("/")
def root():
    return {
        "project": APP_TITLE,
        "endpoints": ["/health", "/predict/xrp (POST JSON)"],
        "expected_features": FEATURE_COLS,
        "model_path": MODEL_PATH
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/xrp")
def predict_xrp(payload: XRPFeatures):
    data = pd.DataFrame([payload.dict()])

    # Extract use_arima flag safely
    if "use_arima" in data.columns:
        use_arima = bool(data.pop("use_arima").iloc[0])
    else:
        use_arima = False

    # Route to ARIMA if requested and available
    if use_arima and arima_model is not None:
        try:
            yhat = float(arima_model.forecast(steps=1)[0])
            return {"predicted_high": yhat, "model": "arima"}
        except Exception as e:
            return {"error": f"ARIMA failed: {e}"}

    # Default XGBoost route
    row = data[FEATURE_COLS]
    yhat = float(model.predict(row)[0])
    return {"predicted_high": yhat, "model": "xgb"}

