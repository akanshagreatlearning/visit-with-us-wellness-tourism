import os
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame, target_col: str = "ProdTaken") -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop CustomerID (identifier)
    - Drop duplicates
    - Convert blank strings to NaN
    - Impute numeric with median, categorical with mode
    """
    df_clean = df.copy()

    # Drop ID-like columns
    if "CustomerID" in df_clean.columns:
        df_clean.drop(columns=["CustomerID"], inplace=True)

    # Drop duplicates
    df_clean = df_clean.drop_duplicates()

    # Normalize empty strings to NaN
    df_clean = df_clean.replace(r"^\s*$", np.nan, regex=True)

    # Identify numeric/categorical columns
    numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()

    # Remove target from feature lists if present
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    # Impute numeric -> median
    for c in numeric_cols:
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())

    # Impute categorical -> mode
    for c in cat_cols:
        mode_val = df_clean[c].mode(dropna=True)
        df_clean[c] = df_clean[c].fillna(mode_val.iloc[0] if len(mode_val) else "Unknown")

    return df_clean


def main():
    # -----------------------
    # Config (env vars + defaults)
    # -----------------------
    REPO_ID = os.getenv("HF_DATASET_REPO_ID", "akanshasalampuria/visit-with-us-wellness-tourism")
    REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")
    RAW_FILENAME = os.getenv("HF_RAW_FILENAME", "tourism.csv")

    TARGET_COL = "ProdTaken"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    PROJECT_DIR = os.getenv("PROJECT_DIR", "tourism_project")
    OUT_DIR = os.path.join(PROJECT_DIR, "data", "processed_xy")
    os.makedirs(OUT_DIR, exist_ok=True)

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in environment. Please set HF_TOKEN before running.")

    # -----------------------
    # Load raw data from Hugging Face Dataset repo
    # -----------------------
    raw_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=RAW_FILENAME
    )

    df = pd.read_csv(raw_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Available columns: {list(df.columns)}")

    print(f"Loaded raw dataset: {df.shape} from HF -> {REPO_ID}/{RAW_FILENAME}")

    # -----------------------
    # Clean
    # -----------------------
    df_clean = clean_data(df, target_col=TARGET_COL)
    print(f"Cleaned dataset: {df_clean.shape}")

    # -----------------------
    # Split into X/y then train/test
    # -----------------------
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # -----------------------
    # Save locally
    # -----------------------
    xtrain_path = os.path.join(OUT_DIR, "Xtrain.csv")
    xtest_path  = os.path.join(OUT_DIR, "Xtest.csv")
    ytrain_path = os.path.join(OUT_DIR, "ytrain.csv")
    ytest_path  = os.path.join(OUT_DIR, "ytest.csv")

    Xtrain.to_csv(xtrain_path, index=False)
    Xtest.to_csv(xtest_path, index=False)
    ytrain.to_csv(ytrain_path, index=False)
    ytest.to_csv(ytest_path, index=False)

    print(f"Saved Xtrain: {Xtrain.shape} -> {xtrain_path}")
    print(f"Saved Xtest : {Xtest.shape} -> {xtest_path}")
    print(f"Saved ytrain: {ytrain.shape} -> {ytrain_path}")
    print(f"Saved ytest : {ytest.shape} -> {ytest_path}")

    # -----------------------
    # Upload back to Hugging Face dataset repo
    # -----------------------
    api = HfApi(token=HF_TOKEN)

    upload_map = {
        xtrain_path: "processed_xy/Xtrain.csv",
        xtest_path:  "processed_xy/Xtest.csv",
        ytrain_path: "processed_xy/ytrain.csv",
        ytest_path:  "processed_xy/ytest.csv",
    }

    for local_path, hf_path in upload_map.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Upload {hf_path}"
        )

    print(f"Uploaded processed X/y splits to HF dataset repo: {REPO_ID}")

    # Verify
    repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print("Repo files (processed_xy):")
    for f in repo_files:
        if f.startswith("processed_xy/"):
            print(" -", f)


if __name__ == "__main__":
    main()
