#!/bin/bash

curl -O https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/nlp_language_modeling/preprocess_data_for_megatron.py

python preprocess_data_for_megatron.py \
    --input train_data.jsonl \
    --json-keys text \
    --tokenizer-library megatron \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file tokenizer/vocab.json \
    --merge-file tokenizer/merges.txt \
    --dataset-impl mmap \
    --output-prefix small_gpt \
    --append-eod
