import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from scripts.evaluate import main

def evaluate_llama32_model():
    print("Starting evaluation for llama-32-11b...")
    target_models = ["llama-32-11b"]
    main(target_models=target_models)
    print("\nFinished evaluation for llama-32-11b.")
    print("Report have been saved.")

if __name__ == "__main__":
    evaluate_llama32_model()