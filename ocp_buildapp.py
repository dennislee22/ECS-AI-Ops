#!/usr/bin/env python3
import subprocess
from pathlib import Path
import time
import os

HOME_DIR = Path.home()
APP_DIR = HOME_DIR
MODELS_DIR = HOME_DIR / "models"
QWEN_MODEL = MODELS_DIR / "Qwen3-8B"
QWEN_GGUF_MODEL = MODELS_DIR / "Qwen3-8B-GGUF"
EMBED_MODEL = MODELS_DIR / "nomic-embed-text-v1.5"
LOG_FILE = HOME_DIR / "cpu-qwen3-app.log"

MODELS_DIR.mkdir(exist_ok=True)

def clone_if_missing(repo_url, target_path):
    if target_path.exists():
        print(f"{target_path} already exists, skipping clone")
        return
    print(f"Cloning {repo_url} into {target_path}...")
    subprocess.run(["git", "clone", repo_url, str(target_path)], check=True)
    subprocess.run(["git", "-C", str(target_path), "lfs", "pull"], check=False)
    print(f"Cloned {repo_url} successfully")

#clone_if_missing("https://huggingface.co/Qwen/Qwen3-8B", QWEN_MODEL)
clone_if_missing("https://huggingface.co/Qwen/Qwen3-8B-GGUF", QWEN_GGUF_MODEL)
clone_if_missing("https://huggingface.co/nomic-ai/nomic-embed-text-v1.5", EMBED_MODEL)

def list_model_files(model_path: Path):
    print(f"\nListing files under {model_path}:")
    if not model_path.exists():
        print(f"  [ERROR] Path does not exist: {model_path}")
        return
    for path in model_path.rglob("*"):
        if path.is_file():
            print(f"  {path.relative_to(HOME_DIR)} (size={path.stat().st_size} bytes)")

list_model_files(QWEN_MODEL)
list_model_files(EMBED_MODEL)

print(f"\nStarting ECS AI Ops from {APP_DIR}/app.py on port 8080...")

cmd = [
    "python",
    str(APP_DIR / "app.py"),
    "--host", "0.0.0.0",
    "--port", "8080",
    "--model-dir", str(QWEN_GGUF_MODEL / "Qwen3-8B-Q4_K_M.gguf"),
    "--embed-dir", str(EMBED_MODEL),
]
with open(LOG_FILE, "w") as f:
    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

print(f"Application started, logging to {LOG_FILE}")

process.wait()
