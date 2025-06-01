import json
from datasets import load_dataset

splits = ['train', 'validation', 'test']
max_samples_per_split = {'train': 1000, 'validation': 200, 'test': 200}

for split in splits:
    dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)
    file_name = f"{split}_data.jsonl"
    
    with open(file_name, "w", encoding="utf-8") as f:
        count = 0
        for sample in dataset:
            if count >= max_samples_per_split[split]:
                break
            text = sample['text'].strip()
            if text:
                json.dump({"text": text}, f)
                f.write("\n")
                count += 1

    print(f"{file_name} created with {count} samples.")
