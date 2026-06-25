import os
import sys

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import importlib
from huggingface_hub import login, HfApi, create_repo
import hf_credentials 

# Force-reload credentials file to capture recent token changes
importlib.reload(hf_credentials)

# Resolve absolute paths and append root directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Configure Hugging Face repository identifiers
HF_USER = hf_credentials.HF_USERNAME
REPO_ID = f"{HF_USER}/tourism-dataset"
LOCAL_DATA_DIR = "tourism_project/data"

# Authenticate session
login(token=hf_credentials.HF_TOKEN)

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
