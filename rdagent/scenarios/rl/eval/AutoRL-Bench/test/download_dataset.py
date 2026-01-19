"""
Download GSM8K dataset to assets/data/gsm8k/

Usage:
    python test/download_dataset.py

Background:
    nohup python test/download_dataset.py > log/dataset_download.log 2>&1 &
"""

from datasets import load_dataset
import json
from pathlib import Path

def main():
    print('Downloading GSM8K dataset...')
    dataset = load_dataset('gsm8k', 'main')

    # Save to assets/data/gsm8k/
    data_dir = Path(__file__).parent.parent / 'assets' / 'data' / 'gsm8k'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save train
    print(f'Saving train split ({len(dataset["train"])} samples)...')
    with open(data_dir / 'train.jsonl', 'w') as f:
        for item in dataset['train']:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save test
    print(f'Saving test split ({len(dataset["test"])} samples)...')
    with open(data_dir / 'test.jsonl', 'w') as f:
        for item in dataset['test']:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print('GSM8K download complete!')
    print(f'Data saved to: {data_dir.absolute()}')

if __name__ == '__main__':
    main()

