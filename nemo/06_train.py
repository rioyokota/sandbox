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

# Setup model using config
gpt_config = llm.GPTConfig(
    num_layers=config.model.num_layers,
    hidden_size=config.model.hidden_size,
    ffn_hidden_size=config.model.ffn_hidden_size,
    num_attention_heads=config.model.num_attention_heads,
    seq_length=config.model.seq_length,
    init_method_std=config.model.init_method_std,
    hidden_dropout=config.model.hidden_dropout,
    attention_dropout=config.model.attention_dropout,
    layernorm_epsilon=config.model.layernorm_epsilon
)

model = llm.GPTModel(gpt_config, tokenizer=tokenizer)

# Setup DataModule from YAML config
paths = OmegaConf.to_container(config.data.paths, resolve=True)
data_module = PreTrainingDataModule(
    paths=paths,
    seq_length=config.data.seq_length,
    micro_batch_size=config.data.micro_batch_size,
    global_batch_size=config.data.global_batch_size,
    split=config.data.split
)

# Setup trainer from YAML config
strategy = nl.MegatronStrategy(
    tensor_model_parallel_size=config.trainer.strategy.tensor_model_parallel_size,
    pipeline_model_parallel_size=config.trainer.strategy.pipeline_model_parallel_size
)

trainer = nl.Trainer(
    devices=config.trainer.devices,
    accelerator=config.trainer.accelerator,
    max_steps=config.trainer.max_steps,
    log_every_n_steps=config.trainer.log_every_n_steps,
    precision=config.trainer.precision,
    strategy=strategy
)

# Start training
trainer.fit(model, datamodule=data_module)
