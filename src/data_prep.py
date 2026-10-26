"""Construcción de datasets y preprocesamiento sin fuga de datos."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DATA_PATH = RAW_DIR / "Ames_Iowa_Housing_Dataset.csv"
INTERIM_DATA_PATH = INTERIM_DIR / "ames_validated.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "ames_modeling.csv"

TARGET = "SalePrice"
ID_COLUMNS = ["Order", "PID"]


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Carga el dataset original sin modificarlo."""
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"El archivo debe incluir la variable objetivo {TARGET!r}.")
    return frame


def build_interim_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Valida el objetivo y elimina filas exactamente duplicadas.

    Este paso no imputa ni escala variables: conserva los datos de la vivienda
    para EDA y deja un registro claro de las correcciones básicas realizadas.
    """
    rows_raw = len(raw)
    valid_target = raw[TARGET].notna() & (raw[TARGET] > 0)
    interim = raw.loc[valid_target].drop_duplicates().copy()
    summary = {
        "filas_raw": rows_raw,
        "filas_sin_target_valido": int((~valid_target).sum()),
        "duplicados_eliminados": int(valid_target.sum() - len(interim)),
        "filas_interim": len(interim),
    }
    return interim, summary


def build_processed_dataset(interim: pd.DataFrame) -> pd.DataFrame:
    """Genera el dataset de modelamiento sin identificadores.

    Conserva los valores faltantes. Imputar, escalar o codificar antes del
    train/test split filtraría información del test hacia el entrenamiento.
    Esas transformaciones se ajustan dentro del pipeline del modelo.
    """
    processed = interim.drop(columns=ID_COLUMNS, errors="ignore").copy()
    if TARGET not in processed.columns:
        raise ValueError(f"El dataset debe incluir {TARGET!r}.")
    return processed


def prepare_project_data() -> dict[str, object]:
    """Construye y guarda las capas interim y processed desde data/raw."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_data()
    interim, summary = build_interim_dataset(raw)
    processed = build_processed_dataset(interim)
    interim.to_csv(INTERIM_DATA_PATH, index=False)
    processed.to_csv(PROCESSED_DATA_PATH, index=False)
    return {"raw": raw, "interim": interim, "processed": processed, "summary": summary}


def load_interim_data() -> pd.DataFrame:
    """Carga la capa validada; la genera si aún no existe."""
    if not INTERIM_DATA_PATH.exists():
        prepare_project_data()
    return pd.read_csv(INTERIM_DATA_PATH)


def load_processed_data() -> pd.DataFrame:
    """Carga la capa apta para modelamiento; la genera si aún no existe."""
    if not PROCESSED_DATA_PATH.exists():
        prepare_project_data()
    return pd.read_csv(PROCESSED_DATA_PATH)


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa la variable objetivo de las características predictoras."""
    if TARGET not in frame.columns:
        raise ValueError(f"El dataset debe incluir {TARGET!r}.")
    return frame.drop(columns=TARGET), frame[TARGET].astype(float)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Crea un preprocesador para ajustar solo con el conjunto de entrenamiento."""
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Sin dato")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])
