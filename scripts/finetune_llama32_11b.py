import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from scripts.train import main

def train_llama32_model():
    print("Starting training for llama-32-11b...")
    target_models = ["llama-32-11b"]
    main(target_models=target_models)
    print("Finished training for llama-32-11b.")

if __name__ == "__main__":
    train_llama32_model()