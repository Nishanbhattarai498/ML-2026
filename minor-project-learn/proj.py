from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = ROOT_DIR / "expiry_dataset.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "expiry_model.joblib"


@dataclass(frozen=True)
class PredictionInput:
	food_type: str
	storage_condition: int
	arrival_date: date
	harvest_date: date


def _parse_date(value: str) -> date:
	try:
		return datetime.strptime(value, "%Y-%m-%d").date()
	except ValueError as exc:
		raise argparse.ArgumentTypeError(
			"Invalid date format. Use YYYY-MM-DD (example: 2026-01-06)."
		) from exc


def load_dataset(csv_path: Path) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(
			f"Dataset not found at {csv_path}. Expected a CSV with training rows."
		)
	df = pd.read_csv(csv_path)

	required_cols = {
		"food_type",
		"storage_condition",
		"days_since_harvest",
		"remaining_useful_life_days",
	}
	missing = required_cols.difference(df.columns)
	if missing:
		raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

	return df


def build_model() -> Pipeline:
	categorical_features = ["food_type"]
	numeric_features = ["storage_condition", "days_since_harvest"]

	preprocessor = ColumnTransformer(
		transformers=[
			(
				"cat",
				OneHotEncoder(handle_unknown="ignore"),
				categorical_features,
			),
			("num", "passthrough", numeric_features),
		]
	)

	model = RandomForestRegressor(
		n_estimators=400,
		random_state=42,
		n_jobs=-1,
	)

	return Pipeline(
		steps=[
			("preprocess", preprocessor),
			("model", model),
		]
	)


def train_and_save(dataset_path: Path, model_path: Path) -> None:
	df = load_dataset(dataset_path)
	X = df.drop(columns=["remaining_useful_life_days"])
	y = df["remaining_useful_life_days"].astype(float)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.25,
		random_state=42,
	)

	pipeline = build_model()
	pipeline.fit(X_train, y_train)

	preds = pipeline.predict(X_test)
	mae = mean_absolute_error(y_test, preds)
	r2 = r2_score(y_test, preds)
	print(f"Test MAE (days): {mae:.2f}")
	print(f"Test R^2: {r2:.3f}")

	model_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(
		{
			"pipeline": pipeline,
			"trained_at": datetime.utcnow().isoformat() + "Z",
			"dataset_path": str(dataset_path),
		},
		model_path,
	)
	print(f"Saved model to: {model_path}")


def _load_model(model_path: Path) -> Pipeline:
	if not model_path.exists():
		raise FileNotFoundError(
			f"Model not found at {model_path}. Run: python proj.py train"
		)
	payload: dict[str, Any] = joblib.load(model_path)
	pipeline = payload.get("pipeline")
	if pipeline is None:
		raise ValueError("Invalid model file: missing 'pipeline'.")
	return pipeline


def predict_shelf_life_days(model_path: Path, input_row: PredictionInput) -> float:
	pipeline = _load_model(model_path)

	days_since_harvest = (input_row.arrival_date - input_row.harvest_date).days
	days_since_harvest = max(0, int(days_since_harvest))

	X = pd.DataFrame(
		[
			{
				"food_type": input_row.food_type,
				"storage_condition": input_row.storage_condition,
				"days_since_harvest": days_since_harvest,
			}
		]
	)
	pred_days = float(pipeline.predict(X)[0])
	return max(0.0, pred_days)


def predict_expiry_date(model_path: Path, input_row: PredictionInput) -> date:
	days = predict_shelf_life_days(model_path, input_row)
	return input_row.arrival_date + timedelta(days=int(round(days)))


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Expiry date predictor (scikit-learn regression)",
	)
	sub = parser.add_subparsers(dest="command", required=True)

	train_p = sub.add_parser("train", help="Train and save the regression model")
	train_p.add_argument(
		"--dataset",
		default=str(DEFAULT_DATASET_PATH),
		help="Path to training CSV dataset",
	)
	train_p.add_argument(
		"--model-out",
		default=str(DEFAULT_MODEL_PATH),
		help="Where to save the trained model (joblib)",
	)

	pred_p = sub.add_parser("predict", help="Predict expiry date for one item")
	pred_p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
	pred_p.add_argument(
		"--food-type",
		required=True,
		help="Food item name/type (e.g., Tomato, Banana, Chicken)",
	)
	pred_p.add_argument(
		"--storage-condition",
		type=int,
		choices=[0, 1],
		required=True,
		help="Room Temp=0, Refrigerated=1",
	)
	pred_p.add_argument(
		"--arrival-date",
		type=_parse_date,
		required=True,
		help="YYYY-MM-DD (date item arrived / inspection date)",
	)
	pred_p.add_argument(
		"--harvest-date",
		type=_parse_date,
		required=True,
		help="YYYY-MM-DD (date item was harvested/produced)",
	)

	args = parser.parse_args()

	if args.command == "train":
		dataset_path = Path(os.path.expanduser(args.dataset)).resolve()
		model_out = Path(os.path.expanduser(args.model_out)).resolve()
		train_and_save(dataset_path, model_out)
		return 0

	if args.command == "predict":
		model_path = Path(os.path.expanduser(args.model)).resolve()
		input_row = PredictionInput(
			food_type=args.food_type,
			storage_condition=args.storage_condition,
			arrival_date=args.arrival_date,
			harvest_date=args.harvest_date,
		)

		pred_days = predict_shelf_life_days(model_path, input_row)
		expiry = predict_expiry_date(model_path, input_row)

		print(f"Predicted shelf life (days): {pred_days:.1f}")
		print(f"Predicted expiry date: {expiry.isoformat()}")
		return 0

	return 2


if __name__ == "__main__":
	raise SystemExit(main())
