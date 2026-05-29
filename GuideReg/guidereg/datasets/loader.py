from pathlib import Path
import pandas as pd

def load_dataset(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: "
            f"{csv_path}"
        )
    return pd.read_csv(csv_path)