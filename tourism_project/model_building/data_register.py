import os
import shutil
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

def main():
    # -----------------------
    # Config
    # -----------------------
    HF_USERNAME = os.getenv("HF_USERNAME", "akanshasalampuria")
    DATASET_NAME = os.getenv("HF_DATASET_NAME", "visit-with-us-wellness-tourism")
    REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"
    REPO_TYPE = "dataset"

    MASTER_DIR = os.getenv("MASTER_DIR", "tourism_project")
    DATA_DIR = os.path.join(MASTER_DIR, "data")
    LOCAL_CSV_SRC = os.getenv("LOCAL_CSV_SRC", "/mnt/data/tourism.csv")  # Colab default upload path
    LOCAL_CSV_NAME = os.getenv("LOCAL_CSV_NAME", "tourism.csv")         # Desired name in DATA_DIR

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment. Set it before running this script.")

    # -----------------------
    # Ensure folders exist
    # -----------------------
    os.makedirs(DATA_DIR, exist_ok=True)

    # -----------------------
    # Ensure CSV is present in data folder
    # -----------------------
    dst_csv_path = os.path.join(DATA_DIR, LOCAL_CSV_NAME)
    if os.path.exists(LOCAL_CSV_SRC) and not os.path.exists(dst_csv_path):
        shutil.copy(LOCAL_CSV_SRC, dst_csv_path)
        print(f"Copied CSV to: {dst_csv_path}")
    elif os.path.exists(dst_csv_path):
        print(f"CSV already exists at: {dst_csv_path}")
    else:
        print("CSV not found at /mnt/data/tourism.csv and not present in tourism_project/data.")
        print("   Please upload the CSV or set LOCAL_CSV_SRC env var.")

    # -----------------------
    # Create/check HF dataset repo
    # -----------------------
    api = HfApi(token=hf_token)

    try:
        api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
        print(f"Dataset repo already exists: {REPO_ID}")
    except RepositoryNotFoundError:
        create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, token=hf_token)
        print(f"Created dataset repo: {REPO_ID}")

    # -----------------------
    # Upload data folder to HF dataset repo
    # -----------------------
    api.upload_folder(
        folder_path=DATA_DIR,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Upload raw tourism dataset"
    )

    print("Upload complete:", REPO_ID)

    # -----------------------
    # Verify upload (lists files in repo)
    # -----------------------
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print("Files in HF dataset repo:")
    for f in files:
        print(" -", f)

if __name__ == "__main__":
    main()
