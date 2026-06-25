import os
import sys

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except:
    current_dir = os.getcwd()

# Get the path of the root directory (one level up)
root_dir = os.path.dirname(current_dir)

# Add root to sys.path if it is not already there
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from huggingface_hub import HfApi, create_repo, login
import os
from hf_credentials import HF_USERNAME, HF_TOKEN

login(HF_TOKEN)
api = HfApi()

create_repo(
    repo_id=f"{HF_USERNAME}/wellness-tourism-prediction",
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True
)

api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id=f"{HF_USERNAME}/wellness-tourism-prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)

print("Deployment files successfully uploaded to Hugging Face Space!")
