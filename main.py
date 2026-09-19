from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import json
import numpy as np
import os

app = FastAPI(title="Smart-Demand API")

# Load model & config
rf_model  = joblib.load("models/random_forest_model.joblib")
lr_model  = joblib.load("models/linear_regression_model.joblib")
label_enc = joblib.load("models/label_encoder.joblib")

with open("models/config.json") as f:
    config = json.load(f)

FEATURES          = config["features"]
CATEGORIES        = config["categories"]
SEASONALITY_INDEX = {int(k): v for k, v in config["seasonality_index"].items()}
MODEL_METRICS     = config["model_metrics"]


# Schemas
class PredictRequest(BaseModel):
    category: str
    avg_price: float
    avg_freight: float
    avg_rating: float
    discount: float
    num_products: float = 50.0
    month: int
    last_month_sales: float
    model: str = "rf"


class ConfigResponse(BaseModel):
    categories: list
    seasonality_index: dict
    model_metrics: dict


# Routes
@app.get("/", response_class=FileResponse)
def serve_frontend():
    return FileResponse("index.html")


@app.get("/api/config", response_model=ConfigResponse)
def get_config():
    return {
        "categories": CATEGORIES,
        "seasonality_index": SEASONALITY_INDEX,
        "model_metrics": MODEL_METRICS,
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.category in label_enc.classes_:
        cat_encoded = int(label_enc.transform([req.category])[0])
    else:
        cat_encoded = 0

    seas_index = SEASONALITY_INDEX.get(req.month, 0.5)

    # price_range: estimate from avg_price (assume 2x spread)
    price_range = req.avg_price * 1.5

    feature_values = {
        'avg_price'         : req.avg_price,
        'discount_rate'     : req.discount / 100,
        'avg_freight'       : req.avg_freight,
        'avg_rating'        : req.avg_rating,
        'num_products'      : req.num_products,
        'price_range'       : price_range,
        'category_encoded'  : cat_encoded,
        'seasonality_index' : seas_index,
        'last_month_sales'  : req.last_month_sales,
    }

    input_vec = np.array([[feature_values[f] for f in FEATURES]])

    if req.model == "lr":
        raw        = float(lr_model.predict(input_vec)[0])
        model_name = "Linear Regression"
        metrics    = MODEL_METRICS["linear_regression"]
        units      = max(0, round(raw))
        dist       = _heuristic_distribution(units)
    else:
        tree_preds = np.array([
            tree.predict(input_vec)[0]
            for tree in rf_model.estimators_
        ])
        raw        = float(tree_preds.mean())
        model_name = "Random Forest"
        metrics    = MODEL_METRICS["random_forest"]
        units      = max(0, round(raw))
        dist       = _rf_tree_distribution(tree_preds)

    # Demand level - adjusted to actual model prediction range
    if units < 70:
        level     = "Low Demand"
        level_key = "low"
    elif units < 120:
        level     = "Moderate Demand"
        level_key = "moderate"
    elif units < 180:
        level     = "High Demand"
        level_key = "high"
    else:
        level     = "Very High Demand"
        level_key = "veryhigh"

    # Score 0-100 based on actual prediction range (max ~300)
    score = min(100, int((units / 300) * 100))

    return {
        "units"             : units,
        "level"             : level,
        "level_key"         : level_key,
        "score"             : score,
        "distribution"      : dist,
        "seasonality_index" : round(seas_index, 4),
        "model_name"        : model_name,
        "metrics"           : metrics,
    }


def _classify_unit(u):
    if u < 70:    return "Low"
    elif u < 120: return "Moderate"
    elif u < 180: return "High"
    else:         return "Very High"


def _rf_tree_distribution(tree_preds: np.ndarray) -> dict:
    counts = {"Low": 0, "Moderate": 0, "High": 0, "Very High": 0}
    for pred in tree_preds:
        counts[_classify_unit(max(0, pred))] += 1
    total = len(tree_preds)
    return {k: round(v / total * 100) for k, v in counts.items()}


def _heuristic_distribution(units: int) -> dict:
    if units < 70:
        return {"Low": 72, "Moderate": 20, "High": 6, "Very High": 2}
    elif units < 120:
        mod = int(40 + (units / 120) * 35)
        low = max(5, 45 - mod)
        hi  = max(5, 100 - mod - low - 3)
        return {"Low": low, "Moderate": mod, "High": hi, "Very High": 3}
    elif units < 180:
        hi  = int(45 + ((units - 120) / 60) * 35)
        mod = max(8, 50 - hi)
        vhi = max(5, 100 - hi - mod - 5)
        return {"Low": 5, "Moderate": mod, "High": hi, "Very High": vhi}
    else:
        vhi = min(85, int(50 + (units - 180) / 120 * 30))
        hi  = max(8, 90 - vhi)
        return {"Low": 2, "Moderate": 5, "High": hi, "Very High": vhi}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)