"""
Download Qwen2.5-3B-Instruct model to assets/models/

Usage:
    python test/download_model.py

Background:
    nohup python test/download_model.py > log/model_download.log 2>&1 &
"""

from huggingface_hub import snapshot_download
from pathlib import Path

def main():
    model_name = 'Qwen/Qwen2.5-3B-Instruct'
    target_dir = Path(__file__).parent.parent / 'assets' / 'models' / 'Qwen2.5-3B-Instruct'

    print(f'Downloading {model_name}...')
    print(f'Target: {target_dir}')

    snapshot_download(
        repo_id=model_name,
        local_dir=str(target_dir),
    )

    print('Model download complete!')
    print(f'Model saved to: {target_dir.absolute()}')

if __name__ == '__main__':
    main()

