#!/bin/bash
# RD-Agent wrapper for AutoRL-Bench

echo "=== RD-Agent ==="
echo "Task: $TASK"
echo "Model: $BASE_MODEL"
echo "Workspace: $WORKSPACE"

cd /Data/home/v-wanyichen/cwy/program/RD-Agent

# 加载 .env 配置
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

# 设置 rdagent 环境变量
export RL_FILE_PATH=$(dirname $(dirname $MODEL_PATH))
export RL_BASE_MODEL=$BASE_MODEL
export RL_BENCHMARK=$TASK

echo "RL_FILE_PATH: $RL_FILE_PATH"

# 运行 rdagent
python -m rdagent.app.rl.loop \
    --base-model "$BASE_MODEL" \
    --benchmark "$TASK" \
    --step-n ${STEP_N:-5} \
    --loop-n ${LOOP_N:-2}

# 提交最终评测
if [ -n "$GRADING_SERVER_URL" ]; then
    echo "Submitting final evaluation..."
    curl -s -X POST "$GRADING_SERVER_URL/submit" \
        -H "Content-Type: application/json" \
        -d "{\"model_path\": \"$OUTPUT_DIR\"}"
fi
