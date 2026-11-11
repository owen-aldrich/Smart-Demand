"""Check notebook layout and forecasting integrity using synthetic CSVs only.

Run: python -m unittest discover -s tests -v
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "03_Final_Model.ipynb"
CSV_FILES = {
    "df_orders": "olist_orders_dataset.csv",
    "df_items": "olist_order_items_dataset.csv",
    "df_products": "olist_products_dataset.csv",
    "df_reviews": "olist_order_reviews_dataset.csv",
    "df_cat_transl": "product_category_name_translation.csv",
}


def synthetic_sources():
    orders, items, products = [], [], []
    categories = ["alpha", "beta", "gamma"]
    for category in categories:
        for product in range(3):
            products.append({
                "product_id": category + str(product),
                "product_category_name": category,
            })
    for month_index, period in enumerate(pd.period_range("2017-01", periods=24, freq="M")):
        for category_index, category in enumerate(categories):
            quantity = 6 + month_index + category_index * 5
            if category == "beta" and month_index in [5, 12, 18]:
                quantity = 0
            if category == "gamma" and month_index < 4:
                quantity = 0
            for unit in range(quantity):
                order_id = f"o_{month_index}_{category}_{unit // 2}"
                if unit % 2 == 0:
                    orders.append({
                        "order_id": order_id,
                        "order_purchase_timestamp": str(
                            period.to_timestamp() + pd.Timedelta(days=unit % 20)
                        ),
                    })
                items.append({
                    "order_id": order_id,
                    "order_item_id": unit % 2 + 1,
                    "product_id": category + str(unit % 3),
                    "price": 20 + category_index * 4 + month_index * 0.2 + unit % 3,
                    "freight_value": 3 + unit % 2,
                })
    reviews = []
    for index, order in enumerate(orders):
        purchased = pd.Timestamp(order["order_purchase_timestamp"])
        reviews.append({
            "order_id": order["order_id"],
            "review_score": 1 + index % 5,
            "review_answer_timestamp": str(purchased + pd.Timedelta(days=1)),
        })
        # A second review answered after the forecast origin must not be averaged in.
        reviews.append({
            "order_id": order["order_id"],
            "review_score": 5,
            "review_answer_timestamp": str(
                (purchased.to_period("M") + 1).to_timestamp() + pd.Timedelta(days=5)
            ),
        })
    return {
        "df_orders": pd.DataFrame(orders),
        "df_items": pd.DataFrame(items),
        "df_products": pd.DataFrame(products),
        "df_reviews": pd.DataFrame(reviews),
        "df_cat_transl": pd.DataFrame({
            "product_category_name": categories,
            "product_category_name_english": categories,
        }),
    }


def run_notebook(sources=None, stop_after=None):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    sources = synthetic_sources() if sources is None else sources
    namespace = {"__name__": "notebook_under_test"}
    with tempfile.TemporaryDirectory(prefix="smartdemand_test_") as folder:
        data_dir = Path(folder) / "dataset"
        model_dir = Path(folder) / "models_leak_free"
        data_dir.mkdir()
        model_dir.mkdir()
        for key, filename in CSV_FILES.items():
            frame = sources[key].copy()
            frame["unused_column"] = "not_loaded"
            frame.to_csv(data_dir / filename, index=False)

        namespace.update(DATA_PATH=str(data_dir) + os.sep, MODEL_PATH=str(model_dir) + os.sep)
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                ast.parse(cell.source)
                if "configuration" in cell.metadata.get("tags", []):
                    continue
                exec(compile(cell.source, str(NOTEBOOK) + ":" + cell.id, "exec"), namespace)
                if cell.id == "imports":
                    namespace["display"] = lambda *args, **kwargs: None
                    namespace["plt"].show = lambda: None
                if cell.id == stop_after:
                    break

        if (model_dir / "config.json").is_file():
            namespace["saved_config"] = json.loads(
                (model_dir / "config.json").read_text(encoding="utf-8")
            )
            namespace["artifact_files"] = {p.name for p in model_dir.iterdir()}
            namespace["reloaded_pipelines"] = {
                name: joblib.load(model_dir / name)
                for name in ["linear_regression_model.joblib", "random_forest_model.joblib"]
            }
            namespace["reloaded_encoder"] = joblib.load(model_dir / "label_encoder.joblib")
        if "plt" in namespace:
            namespace["plt"].close("all")
    return notebook, namespace


class LeakFreeNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook, cls.ns = run_notebook()

    def test_layout_matches_original_notebooks(self):
        headings = [
            cell.source.splitlines()[0]
            for cell in self.notebook.cells
            if cell.cell_type == "markdown" and cell.source.startswith("## ")
        ]
        self.assertEqual(headings, [
            "## 1. Setup & Library Import", "## 2. Load Dataset",
            "## 3. Data Integration", "## 4. Data Cleaning",
            "## 5. Feature Engineering", "## 6. Exploratory Data Analysis (EDA)",
            "## 7. Train-Test Split", "## 8. Model Training",
            "## 9. Evaluation & Comparison", "## 10. Feature Importance",
            "## 11. Save Model", "## Summary",
        ])
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertFalse(cell.outputs)
        self.assertNotIn("STUDY_START", NOTEBOOK.read_text(encoding="utf-8"))

    def test_join_preserves_items_and_rejects_duplicate_keys(self):
        ns = self.ns
        self.assertEqual(len(ns["master"]), len(ns["df_items"]))
        self.assertEqual(ns["df_clean"]["quantity_ordered"].sum(), len(ns["master_history"]))
        self.assertNotIn("unused_column", ns["df_items"].columns)
        bad_sources = synthetic_sources()
        bad_sources["df_orders"] = pd.concat([
            bad_sources["df_orders"], bad_sources["df_orders"].iloc[[0]]
        ])
        with self.assertRaisesRegex(ValueError, "duplikat"):
            run_notebook(bad_sources, stop_after="join-tables")

    def test_calendar_and_exact_three_month_average(self):
        monthly, features = self.ns["df_clean"], self.ns["df_feat"]
        empty = monthly[
            (monthly["category"] == "beta") & (monthly["period"] == pd.Period("2017-06"))
        ].iloc[0]
        self.assertEqual(empty["quantity_ordered"], 0)
        self.assertTrue(pd.isna(empty["avg_price"]))
        lookup = monthly.set_index(["category", "period"])["quantity_ordered"]
        for row in features.itertuples(index=False):
            expected = np.mean([
                lookup.loc[(row.category, row.period - offset)] for offset in [3, 2, 1]
            ])
            self.assertEqual(row.sales_avg_3m, expected)
        self.assertNotIn("last_month_sales", self.ns["FEATURES"])
        self.assertEqual(
            monthly[monthly["category"] == "gamma"]["period"].min(), pd.Period("2017-05")
        )
        self.assertEqual(self.ns["excluded_month"], pd.Period("2018-12"))
        self.assertEqual(monthly["period"].max(), pd.Period("2018-11"))

    def test_gaps_and_future_target_leakage(self):
        ns = self.ns
        monthly = ns["df_clean"]
        with self.assertRaisesRegex(ValueError, "tidak berurutan"):
            ns["build_features"](monthly.drop(monthly.index[5]))
        changed = monthly.copy()
        pivot = pd.Period("2018-03")
        for column in ["quantity_ordered", "avg_price", "avg_freight", "avg_rating", "num_products", "price_range"]:
            mask = changed["period"] >= pivot
            changed.loc[mask, column] = changed.loc[mask, column].fillna(0) + 100_000
        altered = ns["build_features"](changed)
        original = ns["df_feat"]
        columns = ["category", "period", "last_month_sales"] + [
            col for col in ns["NUMERIC_FEATURES"] if col in original.columns
        ]
        pd.testing.assert_frame_equal(
            original[original["period"] <= pivot][columns],
            altered[altered["period"] <= pivot][columns],
        )

    def test_time_split_and_fixed_parameters_match_notebook_02(self):
        ns = self.ns
        self.assertLess(ns["train"]["period"].max(), ns["test"]["period"].min())
        self.assertEqual(len(ns["train"]["period"].unique()), int(len(ns["months"]) * 0.8))
        expected = {
            "n_estimators": 300, "max_depth": 20, "min_samples_split": 5,
            "min_samples_leaf": 2, "max_features": "sqrt", "bootstrap": True,
            "random_state": 42, "n_jobs": -1,
        }
        parameters = ns["rf_model"].named_steps["model"].get_params()
        self.assertEqual({key: parameters[key] for key in expected}, expected)
        self.assertNotIn("validation", ns)
        self.assertNotIn("ParameterGrid", ns)
        self.assertEqual(ns["selected_model_name"], "Random Forest")

    def test_preprocessing_uses_training_not_test(self):
        ns = self.ns
        for model in [ns["lr_model"], ns["rf_model"]]:
            prep = model.named_steps["preprocess"]
            imputer = prep.named_transformers_["numeric"].named_steps["imputer"]
            expected = ns["train"][ns["NUMERIC_FEATURES"]].median().fillna(0).to_numpy()
            np.testing.assert_allclose(imputer.statistics_, expected)
            unfamiliar = ns["X_test"].iloc[[0]].copy()
            unfamiliar["category"] = "never_seen_before"
            self.assertTrue(np.isfinite(model.predict(unfamiliar)).all())
            self.assertNotIn(
                "never_seen_before", prep.named_transformers_["category"].categories_[0]
            )

    def test_late_and_undated_reviews_do_not_leak(self):
        ns = self.ns
        sources = synthetic_sources()
        self.assertIn("prev_avg_rating", ns["FEATURES"])
        self.assertEqual(int(ns["available_review"].sum()), len(sources["df_orders"]))
        purchases = sources["df_orders"].set_index("order_id")["order_purchase_timestamp"]
        purchased = pd.to_datetime(sources["df_reviews"]["order_id"].map(purchases))
        cutoff = (purchased.dt.to_period("M") + 1).dt.to_timestamp()
        late = pd.to_datetime(sources["df_reviews"]["review_answer_timestamp"]) >= cutoff
        sources["df_reviews"].loc[late, "review_score"] = 1
        first_order = sources["df_orders"].iloc[0]
        exact_cutoff = (pd.Timestamp(first_order["order_purchase_timestamp"]).to_period("M") + 1).to_timestamp()
        sources["df_reviews"] = pd.concat([
            sources["df_reviews"],
            pd.DataFrame([
                {"order_id": first_order["order_id"], "review_score": 5,
                 "review_answer_timestamp": str(exact_cutoff)},
                {"order_id": first_order["order_id"], "review_score": 5,
                 "review_answer_timestamp": None},
            ]),
        ])
        _, altered = run_notebook(sources, stop_after="build-features")
        pd.testing.assert_series_equal(ns["df_clean"]["avg_rating"], altered["df_clean"]["avg_rating"])
        pd.testing.assert_series_equal(ns["df_feat"]["prev_avg_rating"], altered["df_feat"]["prev_avg_rating"])

    def test_seasonality_uses_only_training_and_is_global(self):
        ns = self.ns
        self.assertIn("seasonality_index", ns["FEATURES"])
        self.assertNotIn("month_sin", ns["FEATURES"])
        monthly = ns["train"].groupby(ns["train"]["period"].dt.month)[ns["TARGET"]].mean()
        expected = (monthly - monthly.min()) / (monthly.max() - monthly.min())
        pd.testing.assert_series_equal(ns["monthly_norm"], expected)
        for part in [ns["train"], ns["test"]]:
            index_values = part["period"].dt.month.map(expected).fillna(expected.mean())
            np.testing.assert_allclose(part["seasonality_index"], index_values)
            self.assertTrue((part.groupby(part["period"].dt.month)["seasonality_index"].nunique() == 1).all())

    def test_saved_models_and_cell_only_test_result(self):
        ns = self.ns
        self.assertTrue({
            "linear_regression_model.joblib", "random_forest_model.joblib",
            "label_encoder.joblib", "config.json",
        } == ns["artifact_files"])
        self.assertEqual(
            {name for name in ns["artifact_files"] if name.endswith(".joblib")},
            {"linear_regression_model.joblib", "random_forest_model.joblib", "label_encoder.joblib"},
        )
        for filename, original in [
            ("linear_regression_model.joblib", ns["lr_model"]),
            ("random_forest_model.joblib", ns["rf_model"]),
        ]:
            np.testing.assert_allclose(
                original.predict(ns["X_test"]),
                ns["reloaded_pipelines"][filename].predict(ns["X_test"]),
            )
            np.testing.assert_allclose(
                ns["predict_saved_model"](
                    ns["reloaded_pipelines"][filename], ns["test"], ns["loaded_config"]
                ),
                np.maximum(original.predict(ns["X_test"]), 0),
            )
        encoder = ns["reloaded_encoder"]
        original_encoder = ns["rf_model"].named_steps["preprocess"].named_transformers_["category"]
        np.testing.assert_array_equal(
            encoder.transform(ns["test"][["category"]]),
            original_encoder.transform(ns["test"][["category"]]),
        )
        self.assertEqual(ns["saved_config"]["category_encoder_type"], "OneHotEncoder")
        self.assertEqual(
            list(encoder.categories_[0]), sorted(ns["train"]["category"].unique().tolist())
        )
        self.assertEqual(ns["test_result"].shape, (4, 3))
        self.assertEqual(list(ns["test_result"].columns), ["MAE", "RMSE", "R2"])
        self.assertEqual(ns["test_result"].index.name, "Model")
        self.assertNotIn("test_predictions", ns)
        self.assertNotIn("forecast", ns)
        self.assertFalse(any(
            ".to_csv(" in cell.source
            for cell in self.notebook.cells if cell.cell_type == "code"
        ))
        for name in ns["test_result"].index:
            self.assertAlmostEqual(
                ns["test_result"].loc[name, "MAE"],
                ns["saved_config"]["test_metrics"][name]["MAE"], places=8,
            )
        self.assertEqual(ns["saved_config"]["trained_through"], "2018-07")
        self.assertEqual(ns["saved_config"]["selected_model"], ns["selected_model_name"])
        self.assertNotIn("validation_metrics", ns["saved_config"])

    def test_test_targets_cannot_change_trained_models(self):
        sources = synthetic_sources()
        extra_orders, extra_items = [], []
        for month in ["2018-08", "2018-09", "2018-10", "2018-11"]:
            for unit in range(30):
                order_id = f"future_{month}_{unit}"
                extra_orders.append({
                    "order_id": order_id, "order_purchase_timestamp": month + "-10 00:00:00"
                })
                extra_items.append({
                    "order_id": order_id, "order_item_id": 1, "product_id": "alpha0",
                    "price": 9999, "freight_value": 9999,
                })
        sources["df_orders"] = pd.concat([sources["df_orders"], pd.DataFrame(extra_orders)])
        sources["df_items"] = pd.concat([sources["df_items"], pd.DataFrame(extra_items)])
        _, altered = run_notebook(sources, stop_after="train-forest")
        pd.testing.assert_frame_equal(self.ns["X_train"], altered["X_train"])
        pd.testing.assert_series_equal(self.ns["y_train"], altered["y_train"])
        for name in ["lr_model", "rf_model"]:
            np.testing.assert_allclose(
                self.ns[name].predict(self.ns["X_test"]),
                altered[name].predict(self.ns["X_test"]),
            )
        self.assertEqual(self.ns["selected_model_name"], altered["selected_model_name"])


if __name__ == "__main__":
    unittest.main()
