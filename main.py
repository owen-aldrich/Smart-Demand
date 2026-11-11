from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import joblib
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

with (MODEL_DIR / "config.json").open(encoding="utf-8") as file:
    config = json.load(file)

rf_model = joblib.load(MODEL_DIR / "random_forest_model.joblib")
lr_model = joblib.load(MODEL_DIR / "linear_regression_model.joblib")
category_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")

FEATURES = config["features"]
CATEGORIES = config["categories"]
SEASONALITY_INDEX = {
    int(month): float(value) for month, value in config["seasonality_index"].items()
}
SEASONALITY_FALLBACK = float(config["seasonality_fallback"])
TEST_METRICS = config["test_metrics"]
MODEL_NAMES = {"rf": "Random Forest", "lr": "Linear Regression"}
MODELS = {"rf": rf_model, "lr": lr_model}
MODEL_METRICS = {
    key: {
        "test_mae": TEST_METRICS[name]["MAE"],
        "test_rmse": TEST_METRICS[name]["RMSE"],
        "test_r2": TEST_METRICS[name]["R2"],
    }
    for key, name in [
        ("random_forest", "Random Forest"),
        ("linear_regression", "Linear Regression"),
    ]
}

# Pipelines already contain their fitted encoders; never encode the input twice.
if type(category_encoder).__name__ != config["category_encoder_type"]:
    raise RuntimeError("Category encoder does not match config.json.")
if list(category_encoder.categories_[0]) != CATEGORIES:
    raise RuntimeError("Encoder categories do not match config.json.")
for pipeline in MODELS.values():
    if list(pipeline.feature_names_in_) != FEATURES:
        raise RuntimeError("Pipeline feature order does not match config.json.")
    pipeline_categories = pipeline.named_steps["preprocess"].named_transformers_[
        "category"
    ].categories_[0]
    if list(pipeline_categories) != CATEGORIES:
        raise RuntimeError("Pipeline and exported category encoder do not match.")
if config["history_window_months"] != 3:
    raise RuntimeError("This app expects a three-month history window.")

# Existing UI volume bands are illustrative, not learned classification thresholds.
VOLUME_INDEX_REFERENCE = 300
LEVEL_LABELS = {"Low": "low", "Moderate": "moderate", "High": "high", "Very High": "veryhigh"}
app = FastAPI(title="Smart-Demand API", version="2.0")


@app.exception_handler(RequestValidationError)
async def validation_error_response(request, error):
    # Report validation messages without echoing non-JSON values such as NaN.
    return JSONResponse(status_code=422, content={
        "detail": [
            {key: item[key] for key in ["loc", "msg", "type"]}
            for item in error.errors()
        ]
    })


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    category: str = Field(min_length=1, max_length=200)
    # Oldest to newest: t-3, t-2, t-1. These are observed order item counts.
    sales_history: list[int] = Field(min_length=3, max_length=3)
    prev_avg_price: float | None = Field(default=None, ge=0)
    prev_avg_freight: float | None = Field(default=None, ge=0)
    prev_avg_rating: float | None = Field(default=None, ge=1, le=5)
    prev_num_products: int = Field(ge=0)
    prev_price_range: float | None = Field(default=None, ge=0)
    month: int = Field(ge=1, le=12)
    model: Literal["rf", "lr"] = "rf"

    @field_validator("category")
    @classmethod
    def known_category(cls, value):
        value = value.strip()
        if value not in CATEGORIES:
            raise ValueError("Choose a category listed by /api/config.")
        return value

    @field_validator("sales_history")
    @classmethod
    def nonnegative_history(cls, values):
        if any(value < 0 for value in values):
            raise ValueError("Order quantities must be nonnegative.")
        return values

    @model_validator(mode="after")
    def consistent_previous_month(self):
        if self.prev_num_products > self.sales_history[-1]:
            raise ValueError("Unique products cannot exceed last month's ordered units.")
        if self.sales_history[-1] == 0 and any(
            value is not None
            for value in [
                self.prev_avg_price,
                self.prev_avg_freight,
                self.prev_avg_rating,
                self.prev_price_range,
            ]
        ):
            raise ValueError(
                "With zero orders last month, leave price, freight, rating, "
                "and price range empty."
            )
        return self


class ConfigResponse(BaseModel):
    categories: list[str]
    features: list[str]
    seasonality_index: dict[int, float]
    seasonality_fallback: float
    model_metrics: dict
    test_metrics: dict
    target: str
    history_window_months: int
    currency: str
    trained_through: str
    split_summary: list[dict]


def build_feature_frame(request: PredictRequest) -> pd.DataFrame:
    values = {
        "sales_avg_3m": float(np.mean(request.sales_history)),
        "prev_avg_price": request.prev_avg_price,
        "prev_avg_freight": request.prev_avg_freight,
        "prev_avg_rating": request.prev_avg_rating,
        "prev_num_products": request.prev_num_products,
        "prev_price_range": request.prev_price_range,
        "category": request.category,
        "seasonality_index": SEASONALITY_INDEX.get(
            request.month, SEASONALITY_FALLBACK
        ),
    }
    # Match notebook missing-value handling rather than substituting zeros.
    values = {
        key: np.nan if value is None else value for key, value in values.items()
    }
    return pd.DataFrame([values], columns=FEATURES)


def classify_units(units: float) -> str:
    if units < 70:
        return "Low"
    if units < 120:
        return "Moderate"
    if units < 180:
        return "High"
    return "Very High"


def rf_tree_distribution(tree_predictions: np.ndarray) -> dict[str, int]:
    labels = list(LEVEL_LABELS)
    counts = np.array([
        sum(classify_units(max(0, prediction)) == label for prediction in tree_predictions)
        for label in labels
    ])
    percentages = counts / counts.sum() * 100
    rounded = np.floor(percentages).astype(int)
    remainder = 100 - int(rounded.sum())
    order = np.argsort(-(percentages - rounded), kind="stable")
    rounded[order[:remainder]] += 1
    return dict(zip(labels, (int(value) for value in rounded)))


@app.get("/", response_class=FileResponse)
def serve_frontend():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/favicon.ico", response_class=FileResponse, include_in_schema=False)
@app.get("/favicon.svg", response_class=FileResponse, include_in_schema=False)
def serve_favicon():
    return FileResponse(BASE_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/config", response_model=ConfigResponse)
def get_config():
    return {
        "categories": CATEGORIES,
        "features": FEATURES,
        "seasonality_index": SEASONALITY_INDEX,
        "seasonality_fallback": SEASONALITY_FALLBACK,
        "model_metrics": MODEL_METRICS,
        "test_metrics": TEST_METRICS,
        "target": config["target"],
        "history_window_months": config["history_window_months"],
        "currency": "BRL",
        "trained_through": config["trained_through"],
        "split_summary": config["split_summary"],
    }


@app.post("/api/predict")
def predict(request: PredictRequest):
    frame = build_feature_frame(request)
    pipeline = MODELS[request.model]

    try:
        raw = float(pipeline.predict(frame)[0])
        if not np.isfinite(raw):
            raise ValueError("Model returned a non-finite prediction.")
        distribution = None
        if request.model == "rf":
            transformed = pipeline.named_steps["preprocess"].transform(frame)
            tree_predictions = np.array([
                tree.predict(transformed)[0]
                for tree in pipeline.named_steps["model"].estimators_
            ])
            distribution = rf_tree_distribution(tree_predictions)
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail="Input values are outside the model's supported numeric range.",
        ) from error

    raw = max(0, raw)
    units = round(raw)
    level = classify_units(units)
    metrics_key = "random_forest" if request.model == "rf" else "linear_regression"
    return {
        "units": units,
        "raw_units": raw,
        "level": f"{level} Volume",
        "level_key": LEVEL_LABELS[level],
        "score": min(100, int(units / VOLUME_INDEX_REFERENCE * 100)),
        "volume_index_reference": VOLUME_INDEX_REFERENCE,
        "distribution": distribution,
        "distribution_kind": "tree_forecast_spread" if distribution else None,
        "seasonality_index": float(frame["seasonality_index"].iloc[0]),
        "model_name": MODEL_NAMES[request.model],
        "metrics": MODEL_METRICS[metrics_key],
        "sales_avg_3m": float(frame["sales_avg_3m"].iloc[0]),
        "baselines": {
            "last_month": request.sales_history[-1],
            "average_3m": float(frame["sales_avg_3m"].iloc[0]),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
