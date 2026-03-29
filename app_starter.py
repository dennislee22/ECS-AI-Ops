#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# ===============================
# Paths
# ===============================
HOME_DIR = Path.home()             # /opt/app-root/src
APP_DIR = HOME_DIR
MODELS_DIR = HOME_DIR / "models"
QWEN_MODEL = MODELS_DIR / "Qwen3-8B"
EMBED_MODEL = MODELS_DIR / "nomic-embed-text-v1.5"

# ===============================
# Ensure models directory exists
# ===============================
MODELS_DIR.mkdir(exist_ok=True)

# ===============================
# Clone helper (blocking, safe)
# ===============================
def clone_if_missing(repo_url, target_path):
    if target_path.exists():
        print(f"{target_path} already exists, skipping clone")
        return
    print(f"Cloning {repo_url} into {target_path}...")
    subprocess.run(["git", "clone", repo_url, str(target_path)], check=True)
    print(f"Cloned {repo_url} successfully")

# ===============================
# Clone models (waits properly)
# ===============================
clone_if_missing("https://huggingface.co/Qwen/Qwen3-8B", QWEN_MODEL)
clone_if_missing("https://huggingface.co/nomic-ai/nomic-embed-text-v1.5", EMBED_MODEL)

# ===============================
# Start app (PID 1)
# ===============================
APP_PORT = "8080"

print(f"Starting ECS AI Ops from {APP_DIR}/app.py on port {APP_PORT}...")

# Replace current process with app.py
os.execvp("python3", [
    "python3",
    str(APP_DIR / "app.py"),
    "--host", "0.0.0.0",
    "--port", APP_PORT,
    "--model-dir", str(QWEN_MODEL),
    "--embed-dir", str(EMBED_MODEL),
])