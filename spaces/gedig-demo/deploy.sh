#!/bin/bash
# Deploy geDIG demo to Hugging Face Spaces

set -e

SPACE_NAME="gedig-demo"
USERNAME="miyaukaz"

echo "🚀 Deploying geDIG Demo to Hugging Face Spaces"
echo ""

# Check login status
if ! huggingface-cli whoami &>/dev/null; then
    echo "📝 Please login to Hugging Face first:"
    huggingface-cli login
fi

echo ""
echo "Creating Space: $USERNAME/$SPACE_NAME"

# Create the space using Python (more reliable)
PYTHON_BIN="${PYTHON_BIN:-../../.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" << 'EOF'
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
username = "miyaukaz"
space_name = "gedig-demo"
repo_id = f"{username}/{space_name}"

# Create space (ignore if exists)
try:
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="streamlit",
        private=False,
        exist_ok=True
    )
    print(f"✅ Space created/confirmed: {repo_id}")
except Exception as e:
    print(f"Space may already exist: {e}")

# Upload files
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
folder_path = os.getcwd()

print(f"📤 Uploading files from {folder_path}")

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=["deploy.sh", "DEPLOY.md", "*.pyc", "__pycache__"]
)

print(f"")
print(f"🎉 Done! Your demo is at:")
print(f"   https://huggingface.co/spaces/{repo_id}")
EOF

echo ""
echo "🎉 Deployment complete!"
echo "   URL: https://huggingface.co/spaces/$USERNAME/$SPACE_NAME"
