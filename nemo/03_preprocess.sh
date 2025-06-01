#!/bin/bash

curl -O https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/nlp_language_modeling/preprocess_data_for_megatron.py

for split in train validation test
do
  python preprocess_data_for_megatron.py \
      --input "${split}_data.jsonl" \
      --json-keys text \
      --tokenizer-library megatron \
      --tokenizer-type GPT2BPETokenizer \
      --vocab-file tokenizer/vocab.json \
      --merge-file tokenizer/merges.txt \
      --dataset-impl mmap \
      --output-prefix "${split}_gpt" \
      --append-eod
done
