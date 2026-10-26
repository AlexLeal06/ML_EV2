"""Entrenamiento, evaluación y guardado de artefactos del proyecto."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from data_prep import ROOT, build_preprocessor, load_processed_data, prepare_project_data, split_features_target


RANDOM_STATE = 42
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
METRICS_DIR = REPORT_DIR / "metrics"
FIGURES_DIR = REPORT_DIR / "figures"
TABLES_DIR = REPORT_DIR / "tables"
CLUSTER_FEATURES = [
    "Gr Liv Area", "Lot Area", "Overall Qual", "Overall Cond", "Year Built",
    "Bedroom AbvGr", "Full Bath", "Garage Cars",
]


def ensure_output_dirs() -> None:
    """Crea la estructura compacta de resultados del proyecto."""
    for folder in (MODEL_DIR, METRICS_DIR, FIGURES_DIR, TABLES_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Define los modelos supervisados con su propio preprocesamiento."""
    estimators = {
        "Ridge": Ridge(alpha=10.0),
        "Árbol de decisión": DecisionTreeRegressor(
            max_depth=20, min_samples_leaf=5, random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=80, min_samples_leaf=2, max_features=0.8,
            n_jobs=1, random_state=RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline([
            ("preprocessor", build_preprocessor(X)),
            ("estimator", estimator),
        ])
        for name, estimator in estimators.items()
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de regresión en la unidad original: USD."""
    return {
        "MAE_USD": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_USD": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """Selecciona por validación cruzada en train y evalúa una vez en test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    models = build_models(X_train)
    for name, pipeline in models.items():
        scores = -cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        rows.append({
            "Modelo": name,
            "CV_RMSE_USD": float(scores.mean()),
            "CV_RMSE_std_USD": float(scores.std()),
            "MAE_USD": np.nan,
            "RMSE_USD": np.nan,
            "R2": np.nan,
        })

    metrics = pd.DataFrame(rows).sort_values("CV_RMSE_USD").reset_index(drop=True)
    best_name = str(metrics.loc[0, "Modelo"])
    best_model = models[best_name].fit(X_train, y_train)
    test_metrics = regression_metrics(y_test, best_model.predict(X_test))
    metrics.loc[metrics["Modelo"] == best_name, ["MAE_USD", "RMSE_USD", "R2"]] = [
        test_metrics["MAE_USD"],
        test_metrics["RMSE_USD"],
        test_metrics["R2"],
    ]

    baseline = DummyRegressor(strategy="median").fit(
        np.zeros((len(y_train), 1)), y_train,
    )
    baseline_row = {
        "Modelo": "Baseline mediana",
        "CV_RMSE_USD": np.nan,
        "CV_RMSE_std_USD": np.nan,
        **regression_metrics(y_test, baseline.predict(np.zeros((len(y_test), 1)))),
    }
    metrics = pd.concat([metrics, pd.DataFrame([baseline_row])], ignore_index=True)
    return {
        "metrics": metrics,
        "best_name": best_name,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_final_artifacts(X: pd.DataFrame, y: pd.Series, best_name: str) -> None:
    """Ajusta con todos los datos y guarda preprocesador y estimador separados."""
    ensure_output_dirs()
    preprocessor = build_preprocessor(X).fit(X)
    estimator = clone(build_models(X)[best_name].named_steps["estimator"])
    estimator.fit(preprocessor.transform(X), y)
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(estimator, MODEL_DIR / "best_model.joblib")


def cluster_homes(frame: pd.DataFrame, n_clusters: int = 2) -> tuple[pd.DataFrame, float, list[str]]:
    """Aplica K-means a características físicas, excluyendo SalePrice del ajuste."""
    features = [column for column in CLUSTER_FEATURES if column in frame.columns]
    values = SimpleImputer(strategy="median").fit_transform(frame[features])
    scaled = StandardScaler().fit_transform(values)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_STATE)
    labels = model.fit_predict(scaled)
    segmented = frame.copy()
    segmented["Cluster"] = labels
    return segmented, float(silhouette_score(scaled, labels)), features


def _save_model_figure(metrics: pd.DataFrame) -> None:
    plotted = metrics.dropna(subset=["CV_RMSE_USD"]).sort_values("CV_RMSE_USD")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(plotted["Modelo"], plotted["CV_RMSE_USD"], color="#287c8e")
    ax.set(title="Selección por validación cruzada", ylabel="RMSE promedio (USD)")
    ax.tick_params(axis="x", rotation=15)
    ax.bar_label(bars, fmt="$%.0f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_and_save() -> pd.DataFrame:
    """Ejecuta las capas de datos, evalúa modelos y guarda los resultados esenciales."""
    prepare_project_data()
    ensure_output_dirs()
    processed = load_processed_data()
    X, y = split_features_target(processed)
    evaluation = evaluate_models(X, y)
    metrics = evaluation["metrics"]
    metrics.to_csv(METRICS_DIR / "model_metrics.csv", index=False)
    _save_model_figure(metrics)
    save_final_artifacts(X, y, str(evaluation["best_name"]))
    summary = {
        "best_model": evaluation["best_name"],
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "cv_folds": 5,
        "selection_metric": "CV_RMSE_USD",
        "rows": len(processed),
        "predictors": X.shape[1],
    }
    (METRICS_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(metrics.round({"MAE_USD": 0, "RMSE_USD": 0, "R2": 3}).to_string(index=False))
    return metrics


if __name__ == "__main__":
    train_and_save()
