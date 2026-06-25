import os
from huggingface_hub import login, HfApi, create_repo

HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")

REPO_ID = f"{HF_USERNAME}/tourism-dataset"
LOCAL_DATA_DIR = "tourism_project/data"

# Authenticate session
login(token=HF_TOKEN)

# Initialize or verify target dataset repository
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False
    )
    print(f"Repository '{REPO_ID}' created successfully.")
except Exception as e:
    if "RepositoryAlreadyExistsError" in str(e):
        print(f"Notice: Repository '{REPO_ID}' already exists. Proceeding...")
    else:
        print(f"Alert: Failed to create repository: {e}")

# Bulk upload local data directory to Hugging Face Hub
try:
    hf_api = HfApi()
    hf_api.upload_folder(
        folder_path=LOCAL_DATA_DIR,
        repo_id=REPO_ID,  
        repo_type="dataset",
    )
    print(f"Success: Content from '{LOCAL_DATA_DIR}' synced to Hugging Face!")
except Exception as e:
    print(f"Error: Directory upload failed: {e}")
