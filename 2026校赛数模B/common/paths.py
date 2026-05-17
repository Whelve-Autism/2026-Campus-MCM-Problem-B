from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _first_csv(pattern: str) -> Path:
    hits = sorted(DATA_DIR.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No CSV matched {pattern!r} under {DATA_DIR}")
    return hits[0]


MANIFEST_CSV = _first_csv("*molecular_interaction_manifest.csv")
EDGES_CSV = _first_csv("*knn_graph_edges.csv")
