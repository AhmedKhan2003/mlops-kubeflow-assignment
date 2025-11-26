# scripts/get_dataset.py
import pandas as pd
from sklearn.datasets import fetch_openml, fetch_california_housing

def save_boston_or_fallback(out_path="data/raw_data.csv"):
    try:
        b = fetch_openml(name="Boston", as_frame=True)
        df = b.frame
        # some OpenML versions call the target 'MEDV' or 'target' - normalize to MEDV
        if 'MEDV' not in df.columns and 'target' in df.columns:
            df = df.rename(columns={'target': 'MEDV'})
        print("Loaded Boston from OpenML, shape:", df.shape)
    except Exception as e:
        print("Boston unavailable on OpenML — falling back to California housing.", e)
        cal = fetch_california_housing(as_frame=True)
        df = cal.frame
        # create MEDV column to be consistent with Boston naming
        df['MEDV'] = cal.target
        print("Loaded California housing, shape:", df.shape)

    df.to_csv(out_path, index=False)
    print("Saved dataset to", out_path)

if __name__ == "__main__":
    save_boston_or_fallback()
