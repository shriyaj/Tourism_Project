import os
from huggingface_hub import HfApi, create_repo, login
import os

HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")


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
