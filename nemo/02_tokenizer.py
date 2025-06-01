from transformers import GPT2TokenizerFast
import os

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
os.makedirs("tokenizer", exist_ok=True)
tokenizer.save_pretrained("tokenizer")

print("GPT-2 tokenizer downloaded and saved to ./tokenizer.")
