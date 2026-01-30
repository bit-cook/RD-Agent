#!/bin/bash
set -x

DIR="$(
  cd "$(dirname "$(readlink -f "$0")")" || exit
  pwd -P
)"

uv venv --python=3.10.8 --clear $DIR/.venv/

source $DIR/.venv/bin/activate
uv pip install pip
uv pip install -e .
uv pip install -r $DIR/requirements.txt
export MASTER_PORT=$((10000 + RANDOM % 40000))
# source ~/.bashrc
# conda activate reward_env
# cd /home/v-lijingyuan/train_reward
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT \
  train.py \
  --json_path all_pairs_merged.json \
  --val_json_path valid.json \
  --model_name Qwen/Qwen3-4B \
  --batch_size 6 \
  --max_epochs 4 \
  --lr 1e-5 \
  --max_length 2300 \
  --num_workers 4 \
  --log_dir ./logs/rm_1219 \
  --accelerator gpu --devices 8 \
  --precision 16-mixed \
  --loss_type bt --tau 1.0

export CONTAINER_NAME="rdagent"
export STORAGE_ACCOUNT_NAME="epeastus"
export SAS_TOKEN=$(az storage container generate-sas --as-user --auth-mode login --account-name $STORAGE_ACCOUNT_NAME --name $CONTAINER_NAME --permissions lrw --expiry $(date -u -d "6 day" '+%Y-%m-%dT%H:%MZ') -o tsv)
export SAS_TOKEN="2026-01-12T00%3A00Z&sp=rwdl&sv=2022-11-02&sr=c&skoid=8873fcee-b5c9-4fec-9d85-4b5b460ab7eb&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2026-01-05T11%3A24%3A56Z&ske=2026-01-12T00%3A00%3A00Z&sks=b&skv=2022-11-02&sig=PaZxobBpbOGZUUGLDRRUqficOrscJPBqRs/KOeavHj0%3D"
azcopy copy ./logs/last_run_3303 "https://epeastus.blob.core.windows.net/rdagent/FinetuneAgenticLLM/reward_ckpt/?$SAS_TOKEN" --recursive=true
cp -r ./logs/last_run_3303 ~/auto_sync_back/