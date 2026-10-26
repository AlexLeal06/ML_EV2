"""Predicciones con el estimador y preprocesador guardados por separado."""
from pathlib import Path

import joblib
import pandas as pd

try:  # Permite usar `python src/predict.py` y `from src.predict import ...`.
    from .data_prep import ID_COLUMNS, ROOT, TARGET
except ImportError:
    from data_prep import ID_COLUMNS, ROOT, TARGET


MODEL_DIR = ROOT / "models"


def make_prediction(
    houses: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR,
) -> pd.Series:
    """Estima SalePrice para filas nuevas usando el mismo encoder y scaler.

    Acepta un DataFrame con columnas predictoras del dataset original. Si falta
    alguna, se completa como nula para aplicar la imputación aprendida. Se
    ignoran columnas adicionales (por ejemplo SalePrice, PID u Order).
    """
    folder = Path(model_dir)
    preprocessor = joblib.load(folder / "preprocessor.joblib")
    estimator = joblib.load(folder / "best_model.joblib")
    expected = list(preprocessor.feature_names_in_)
    features = houses.drop(columns=[TARGET, *ID_COLUMNS], errors="ignore")
    features = features.reindex(columns=expected)
    predictions = estimator.predict(preprocessor.transform(features))
    return pd.Series(predictions, index=houses.index, name="SalePrice_predicho")
