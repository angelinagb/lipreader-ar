"""
Carga de frases y split train / val / test.

El split es EXACTAMENTE el del notebook 05 (mismo parquet, misma
normalización, GroupShuffleSplit con random_state 18 y 19). Así:
- el diccionario se arma solo con frases de TRAIN, y
- el corrector se evalúa con frases de videos que el modelo nunca vio.
Si el diccionario incluyera frases de test, el corrector "conocería las
respuestas" y los resultados quedarían inflados.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .texto import normalize_text

REPO = Path(__file__).resolve().parent.parent
PARQUET_POR_DEFECTO = REPO / "Modelo" / "notebooks" / "data" / "clips_metadata.parquet"


def cargar_splits(parquet=PARQUET_POR_DEFECTO):
    """Devuelve un dict con los DataFrames 'train', 'val' y 'test'."""
    parquet = Path(parquet)
    if not parquet.exists():
        raise FileNotFoundError(
            f"No encuentro {parquet}. Tiene que ser el clips_metadata.parquet "
            "nuevo (19.618 clips) que genera el notebook 04."
        )

    clips_df = pd.read_parquet(parquet)
    clips_df["label"] = clips_df["label"].astype(str).apply(normalize_text)
    clips_df = clips_df[clips_df["label"].str.len() > 0].reset_index(drop=True)

    train_val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=18)
        .split(clips_df, groups=clips_df["video_id"])
    )
    train_val_df = clips_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = clips_df.iloc[test_idx].reset_index(drop=True)

    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=19)
        .split(train_val_df, groups=train_val_df["video_id"])
    )
    return {
        "train": train_val_df.iloc[train_idx].reset_index(drop=True),
        "val": train_val_df.iloc[val_idx].reset_index(drop=True),
        "test": test_df,
    }
