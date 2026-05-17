from __future__ import annotations

import pandas as pd

from common.paths import EDGES_CSV, MANIFEST_CSV


def load_manifest() -> pd.DataFrame:
    """Attachment 1: row 0 English headers, row 1 descriptions (skipped); data from row 2."""
    df = pd.read_csv(MANIFEST_CSV, skiprows=[1], encoding="utf-8")
    numeric_cols = [
        "Manifold_X",
        "Manifold_Y",
        "Bioactivity_Score",
        "LogP",
        "TPSA",
        "MolWt",
        "Dipole_Proxy",
        "Max_Partial_Charge",
        "Balaban_J",
        "Bertz_CT",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ID"] = df["ID"].astype(int)
    return df


def load_edges() -> pd.DataFrame:
    """Attachment 2; skip Chinese description row."""
    df = pd.read_csv(EDGES_CSV, skiprows=[1], encoding="utf-8")
    df["Source"] = df["Source"].astype(int)
    df["Target"] = df["Target"].astype(int)
    df["Tanimoto_Similarity"] = pd.to_numeric(df["Tanimoto_Similarity"], errors="coerce")
    return df.dropna(subset=["Source", "Target", "Tanimoto_Similarity"])


def manifest_phys_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"ID", "SMILES", "Manifold_X", "Manifold_Y", "Bioactivity_Score"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
