#!/bin/bash
# OpenHands Agent wrapper for AutoRL-Bench

echo "=== OpenHands Agent ==="
echo "Task: $TASK"
echo "Model: $BASE_MODEL"
echo "Workspace: $WORKSPACE"
echo "Grading Server: $GRADING_SERVER_URL"
echo "Output Dir: $OUTPUT_DIR"

# 加载 .env 配置（包含 LLM API Key）
cd /Data/home/v-wanyichen/cwy/program/RD-Agent
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
fi

# 映射环境变量（rdagent 用 OPENAI_API_KEY，openhands 用 LLM_API_KEY）
export LLM_API_KEY="${OPENAI_API_KEY}"
export LLM_MODEL="${CHAT_MODEL:-gpt-4o}"

# 激活 openhands 环境
source ~/cwy/miniconda3/bin/activate openhands

# 运行 openhands-rl
cd /Data/home/v-wanyichen/cwy/program/cwy/openhands-rl

python main.py \
    --benchmark "$TASK" \
    --base-model "$BASE_MODEL" \
    --workspace "$WORKSPACE" \
    --max-iterations ${MAX_ITERATIONS:-50}
