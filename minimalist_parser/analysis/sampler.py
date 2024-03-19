"""
samples entries from a corpus by sampling lines from a file
"""
import os
import random

# usually should be dev set
original_file_path = "../../data/processed/seq2seq/official/train/train.tsv"
new_file_path = "../../data/processed/seq2seq/sampled/official"
os.makedirs(new_file_path, exist_ok=True)
new_file_path += "/conj_from_train_to_predict.tsv"
sample_size = 100

with open(original_file_path, 'r') as original_file:
    lines = original_file.readlines()
    # find the atb and excorporation cases
    lines = [line for line in lines if "hm_atb" in line or "excorporation" in line]
    sample_size = min(sample_size, len(lines))
    print(f"sample {sample_size} lines from the original {len(lines)}")
    sample = random.sample(lines, sample_size)

with open(new_file_path, 'w') as write_file:
    for line in sample:
        write_file.write(line)

