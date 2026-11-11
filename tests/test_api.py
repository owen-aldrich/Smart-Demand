"""Integration checks against the real, user-supplied notebook 03 artifacts."""
import json
import unittest

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

import main


class ForecastApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        cls.payload = {
            "category": "electronics",
            "sales_history": [80, 90, 100],
            "prev_avg_price": 100.0,
            "prev_avg_freight": 20.0,
            "prev_avg_rating": 4.0,
            "prev_num_products": 50,
            "prev_price_range": 55.0,
            "month": 5,
        }

    def test_config_and_frontend_use_new_artifact_schema(self):
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["target"], "quantity_ordered")
        self.assertEqual(result["currency"], "BRL")
        self.assertEqual(result["features"], main.FEATURES)
        self.assertIn("electronics", result["categories"])
        self.assertEqual(result["history_window_months"], 3)
        self.assertEqual(result["trained_through"], "2018-03")
        self.assertAlmostEqual(
            result["model_metrics"]["random_forest"]["test_mae"],
            main.TEST_METRICS["Random Forest"]["MAE"],
        )
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        for element in ["sales1", "sales2", "sales3", "numProducts", "priceRange"]:
            self.assertIn('id="' + element + '"', page.text)
        self.assertNotIn('id="discount"', page.text)
        self.assertNotIn("IDR_TO_BRL", page.text)

    def expected_frame(self, payload):
        # Explicit notebook feature values, independent of the API feature helper.
        return pd.DataFrame([{
            "sales_avg_3m": sum(payload["sales_history"]) / 3,
            "prev_avg_price": payload["prev_avg_price"],
            "prev_avg_freight": payload["prev_avg_freight"],
            "prev_avg_rating": np.nan if payload["prev_avg_rating"] is None else payload["prev_avg_rating"],
            "prev_num_products": payload["prev_num_products"],
            "prev_price_range": payload["prev_price_range"],
            "category": payload["category"],
            "seasonality_index": main.config["seasonality_index"][str(payload["month"])],
        }], columns=main.FEATURES)

    def test_both_models_match_direct_notebook_pipeline_predictions(self):
        for model_key, model_name in [("rf", "Random Forest"), ("lr", "Linear Regression")]:
            with self.subTest(model=model_key):
                payload = {**self.payload, "model": model_key}
                expected = max(0, float(main.MODELS[model_key].predict(self.expected_frame(payload))[0]))
                response = self.client.post("/api/predict", json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertAlmostEqual(result["raw_units"], expected, places=10)
                self.assertEqual(result["units"], round(expected))
                self.assertEqual(result["model_name"], model_name)
                self.assertEqual(result["baselines"], {"last_month": 100, "average_3m": 90.0})
                self.assertEqual(result["seasonality_index"], main.SEASONALITY_INDEX[5])
                if model_key == "rf":
                    self.assertEqual(sum(result["distribution"].values()), 100)
                    self.assertEqual(result["distribution_kind"], "tree_forecast_spread")
                else:
                    self.assertIsNone(result["distribution"])

    def test_missing_rating_uses_fitted_training_imputer(self):
        for model in ["rf", "lr"]:
            payload = {**self.payload, "model": model, "prev_avg_rating": None}
            expected = max(0, float(main.MODELS[model].predict(self.expected_frame(payload))[0]))
            response = self.client.post("/api/predict", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertAlmostEqual(response.json()["raw_units"], expected, places=10)

    def test_price_range_and_product_count_are_actual_inputs(self):
        request = main.PredictRequest(**self.payload)
        frame = main.build_feature_frame(request)
        self.assertEqual(frame["prev_price_range"].iloc[0], 55.0)
        self.assertEqual(frame["prev_num_products"].iloc[0], 50)
        self.assertEqual(frame["sales_avg_3m"].iloc[0], 90)
        self.assertNotIn("last_month_sales", frame.columns)
        self.assertEqual(frame["category"].iloc[0], "electronics")
        # No manually encoded category or artificial price * 1.5 spread.
        self.assertNotIn("category_encoded", frame.columns)

    def test_zero_sales_month_preserves_missing_transaction_conditions(self):
        payload = {
            **self.payload,
            "sales_history": [0, 0, 0],
            "prev_num_products": 0,
            "prev_avg_price": None,
            "prev_avg_freight": None,
            "prev_avg_rating": None,
            "prev_price_range": None,
        }
        frame = main.build_feature_frame(main.PredictRequest(**payload))
        self.assertTrue(frame[["prev_avg_price", "prev_avg_freight", "prev_avg_rating", "prev_price_range"]].isna().all().all())
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreaterEqual(response.json()["raw_units"], 0)

    def test_invalid_and_old_inputs_are_rejected(self):
        for changes in [
            {"category": "not_a_trained_category"}, {"model": "invalid"},
            {"month": 0}, {"month": 13}, {"sales_history": [1, 2]},
            {"sales_history": [-1, 90, 100]}, {"sales_history": [1.2, 90, 100]},
            {"prev_avg_rating": 6}, {"prev_avg_price": -1},
            {"prev_num_products": 101}, {"prev_price_range": -1},
            {"discount": 10}, {"sales_history": [0, 0, 0], "prev_num_products": 0},
        ]:
            with self.subTest(changes=changes):
                response = self.client.post("/api/predict", json={**self.payload, **changes})
                self.assertEqual(response.status_code, 422, response.text)

    def test_nonfinite_inputs_are_rejected(self):
        for value in [float("nan"), float("inf")]:
            with self.subTest(value=value):
                payload = {**self.payload, "prev_avg_price": value}
                response = self.client.post(
                    "/api/predict", content=json.dumps(payload),
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(response.status_code, 422, response.text)


if __name__ == "__main__":
    unittest.main()
