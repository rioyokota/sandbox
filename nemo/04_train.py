import os
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.llm.gpt.data import PreTrainingDataModule

# Tokenizer setup
vocab_path = "tokenizer/vocab.json"
merges_path = "tokenizer/merges.txt"

tokenizer = get_nmt_tokenizer(
    library="megatron",
    model_name="GPT2BPETokenizer",
    vocab_file=vocab_path,
    merges_file=merges_path
)
print(f"Tokenizer initialized (vocab size = {tokenizer.vocab_size}).")

# Tiny GPT config
gpt_config = llm.GPTConfig(
    num_layers=2,
    hidden_size=64,
    ffn_hidden_size=256,
    num_attention_heads=4,
    seq_length=128,
    init_method_std=0.02,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    layernorm_epsilon=1e-5
)

model = llm.GPTModel(gpt_config, tokenizer=tokenizer)

data_module = PreTrainingDataModule(
    paths={
        "train": ["small_gpt_text_document"],
        "validation": ["small_gpt_text_document"],
        "test": ["small_gpt_text_document"]
    },
    seq_length=128,
    micro_batch_size=2,
    global_batch_size=2,
    split="90,10,0"
)

strategy = nl.MegatronStrategy(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1
)

trainer = nl.Trainer(
    devices=1, accelerator="gpu",
    max_steps=100,
    log_every_n_steps=10,
    precision=32,
    strategy=strategy
)

print("Starting training...")
trainer.fit(model, datamodule=data_module)
print("Training completed.")
