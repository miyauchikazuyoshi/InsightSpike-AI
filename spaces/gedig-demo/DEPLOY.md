# Deploying to Hugging Face Spaces

## Prerequisites

1. Hugging Face account
2. `huggingface_hub` CLI installed: `pip install huggingface-cli`
3. Login: `huggingface-cli login`

## Option 1: Using the Web Interface

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Settings:
   - **Owner**: your username
   - **Space name**: `gedig-demo`
   - **License**: Apache 2.0
   - **SDK**: Gradio
4. Upload files:
   - `app_gradio.py`
   - `requirements.txt`
   - `README.md`
5. The Space will automatically build and deploy

## Option 2: Using Git

```bash
# Clone the Space (after creating it on the web)
git clone https://huggingface.co/spaces/YOUR_USERNAME/gedig-demo
cd gedig-demo

# Copy files
cp /path/to/InsightSpike-AI/spaces/gedig-demo/* .

# Push
git add .
git commit -m "Initial deployment"
git push
```

## Option 3: Using huggingface_hub

```python
from huggingface_hub import HfApi

api = HfApi()

# Create Space
api.create_repo(
    repo_id="YOUR_USERNAME/gedig-demo",
    repo_type="space",
    space_sdk="gradio",
    private=False
)

# Upload files
api.upload_folder(
    folder_path="spaces/gedig-demo",
    repo_id="YOUR_USERNAME/gedig-demo",
    repo_type="space"
)
```

## After Deployment

1. Check the Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/gedig-demo`
2. Wait for build to complete (1-2 minutes)
3. Test the demo
4. Share the URL!

## Updating

Just push new changes to the repo:

```bash
git add .
git commit -m "Update demo"
git push
```

The Space will automatically rebuild.

## Troubleshooting

- **Build fails**: Check the logs in the "Logs" tab
- **App crashes**: Ensure all imports are in `requirements.txt`
- **Slow startup**: Gradio builds on first run

## Expected URL

After deployment:
```
https://huggingface.co/spaces/miyaukaz/gedig-demo
```
