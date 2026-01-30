export MASTER_PORT=$((10000 + RANDOM % 40000))
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 --master_port=30234 \
    train.py \
    --json_path all_pairs_merged.json \
    --model_name Qwen/Qwen3-4B \
    --batch_size 6 \
    --max_epochs 5 \
    --lr 1e-5 \
    --max_length 2300 \
    --num_workers 4 \
    --log_dir ./logs/rm_1219 \
    --accelerator gpu --devices 1 \
    --precision 16-mixed --loss_type bt --tau 1.0
