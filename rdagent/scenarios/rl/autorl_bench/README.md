# AutoRL-Bench

LLM Post-Training 强化学习评测基准。

## 快速开始

```bash
cd /Data/home/v-wanyichen/cwy/program/RD-Agent

# 后台运行
nohup python -m rdagent.scenarios.rl.autorl_bench.run_agent \
    --agent-id example_agent \
    --task gsm8k \
    --base-model Qwen/Qwen2.5-0.5B \
    --timeout 7200 \
    > rdagent/scenarios/rl/autorl_bench/log/run.log 2>&1 &

# 查看日志
tail -f rdagent/scenarios/rl/autorl_bench/log/run.log

# 停止
kill $(pgrep -f "run_agent.*example_agent")
```

## 目录结构

```
autorl_bench/
├── run_agent.py          # 入口
├── conf.py               # 配置
├── agents/               # Agent 实现
├── environment/          # 评测环境
├── tasks/                # 任务定义
├── results/              # 运行结果
└── workspace/            # 工作目录
```

## 环境变量

```bash
export AUTORL_FILE_PATH=/path/to/data
```
