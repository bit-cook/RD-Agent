# Example Agent

预写好训练代码，在 Docker 里运行（和真实 workflow 一样）

## 运行

```bash
# 从项目根目录运行
python rdagent/scenarios/rl/eval/autorl_bench/example_agent/main.py \
    --base-model Qwen2.5-Coder-0.5B-Instruct \
    --benchmark gsm8k \
    --train-ratio 0.01 \
    --eval-limit 50
```

## 流程

1. 创建 workspace，注入预写好的 GRPO 训练代码
2. 在 Docker 里运行训练（和真实 workflow 一样）
3. 运行 benchmark 评测
