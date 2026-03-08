import os
from huggingface_hub import HfApi, create_repo

local_dir = "/home/luwa/Documents/pylate/dataset"
repo_id = "lumos2548/pylate-dataset"

api = HfApi()

create_repo(repo_id, repo_type="dataset", exist_ok=True)

api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="dataset",
)

print(f"Successfully uploaded dataset to https://huggingface.co/datasets/{repo_id}")
