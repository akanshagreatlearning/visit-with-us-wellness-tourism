import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

def main():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found. Set HF_TOKEN in environment.")

    hf_username = os.getenv("HF_USERNAME", "akanshasalampuria")
    space_name = os.getenv("HF_SPACE_NAME", "visit-with-us-wellness-app")
    space_repo_id = f"{hf_username}/{space_name}"

    api = HfApi(token=hf_token)

    # Create Space if not exists
    try:
        api.repo_info(repo_id=space_repo_id, repo_type="space")
        print(f"Space exists: {space_repo_id}")
    except RepositoryNotFoundError:
        create_repo(
            repo_id=space_repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=hf_token
        )
        print(f"Created Docker Space: {space_repo_id}")

    base_dir = "tourism_project/deployment"
    files_to_upload = {
        f"{base_dir}/app.py": "app.py",
        f"{base_dir}/requirements.txt": "requirements.txt",
        f"{base_dir}/Dockerfile": "Dockerfile",
    }

    for local_path, repo_path in files_to_upload.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=space_repo_id,
            repo_type="space",
            commit_message=f"Deploy {repo_path}"
        )
        print(f"Uploaded: {repo_path}")

    print("\nDeployment pushed to HF Space:", space_repo_id)

if __name__ == "__main__":
    main()
