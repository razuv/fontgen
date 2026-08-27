from pathlib import Path

from transformers import AutoModel, AutoTokenizer

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TARGET = Path(__file__).parents[1] / "models" / "paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(TARGET)
    model.save_pretrained(TARGET, safe_serialization=True)
    print(f"Saved {MODEL_ID} to {TARGET}")


if __name__ == "__main__":
    main()
