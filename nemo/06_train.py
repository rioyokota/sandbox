from omegaconf import OmegaConf
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.llm.gpt.data import PreTrainingDataModule

# Load configuration from YAML
config = OmegaConf.load("05_train.yaml")

# Setup tokenizer
tokenizer_cfg = config.model.tokenizer
tokenizer = get_nmt_tokenizer(
    library=tokenizer_cfg.library,
    model_name=tokenizer_cfg.type,
    vocab_file=tokenizer_cfg.vocab_file,
    merges_file=tokenizer_cfg.merge_file
)

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
        "train": ["train_gpt_text_document"],
        "validation": ["validation_gpt_text_document"],
        "test": ["test_gpt_text_document"]
    },
    seq_length=128,
    micro_batch_size=2,
    global_batch_size=2,
    split="100,100,100"
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
