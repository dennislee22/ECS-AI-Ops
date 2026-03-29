import os
import subprocess
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
PORT = "9000"
APP_PATH = Path.home() / "ECS-AI-Ops/app.py"
MODELS_DIR = Path.home() / "models"
QWEN_DIR = MODELS_DIR / "Qwen3-8B"
EMBED_DIR = MODELS_DIR / "nomic-embed-text-v1.5"
LOG_FILE = Path.home() / "cpu-qwen3-app.log"

QWEN_REPO = "https://huggingface.co/Qwen/Qwen3-8B"
EMBED_REPO = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5"

# -----------------------------
# Ensure working directory
# -----------------------------
os.chdir(str(Path.home() / "ECS-AI-Ops"))

# -----------------------------
# Create models directory
# -----------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Clone repos if missing (wait for completion)
# -----------------------------
def clone_if_missing(repo_url, path):
    if not path.exists():
        print(f"Cloning {repo_url} into {path}")
        subprocess.run(["git", "clone", repo_url, str(path)], check=True)
        if not path.exists():
            raise RuntimeError(f"Failed to clone {repo_url}")
    else:
        print(f"{path} already exists, skipping clone")

clone_if_missing(QWEN_REPO, QWEN_DIR)
clone_if_missing(EMBED_REPO, EMBED_DIR)

# -----------------------------
# Install Python dependencies
# -----------------------------
requirements_file = Path.home() / "ECS-AI-Ops/requirements.txt"
if requirements_file.exists():
    print(f"Installing Python dependencies...")
    subprocess.run(["pip", "install", "--no-cache-dir", "-r", str(requirements_file)], check=True)

# -----------------------------
# Start the app
# -----------------------------
print(f"Starting app on port {PORT}, logging to {LOG_FILE}")
with open(LOG_FILE, "w") as f:
    subprocess.run([
        "python", str(APP_PATH),
        "--host", "0.0.0.0",
        "--port", PORT,
        "--model-dir", str(QWEN_DIR),
        "--embed-dir", str(EMBED_DIR)
    ], stdout=f, stderr=subprocess.STDOUT)
