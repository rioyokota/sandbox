import json

sample_texts = [
    "NeMo is a toolkit for building AI models. It includes tools for training large language models.",
    "GPT models are transformer-based neural networks that predict the next token in a sequence.",
    "This is a small example dataset for demonstrating GPT training on a single GPU."
]

with open("train_data.jsonl", "w", encoding="utf-8") as f:
    for line in sample_texts:
        json.dump({"text": line}, f)
        f.write("\n")

print("train_data.jsonl created successfully.")
