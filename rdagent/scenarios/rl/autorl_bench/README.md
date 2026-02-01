# AutoRL-Bench

LLM Post-Training 强化学习评测基准（bench）。  
注意：AutoRL-Bench 是**独立的 benchmark**，RD-Agent 是后续要优化的 agent。本阶段以 benchmark 为主。


## 快速开始

```bash
cd /Data/home/v-wanyichen/cwy/program/RD-Agent
pip install -e .
```

### Example Agent（GRPO 训练）

```bash
nohup python -m rdagent.scenarios.rl.autorl_bench.run_agent \
    --agent-id example_agent \
    --task gsm8k \
    --base-model Qwen/Qwen2.5-Coder-0.5B \
    --timeout 7200 \
    > rdagent/scenarios/rl/autorl_bench/log/example_agent.log 2>&1 &

tail -f rdagent/scenarios/rl/autorl_bench/log/example_agent.log
kill $(pgrep -f "run_agent.*example_agent")
```

### RD-Agent（自动假设 + 代码生成）

```bash
nohup python -m rdagent.scenarios.rl.autorl_bench.run_agent \
    --agent-id rdagent \
    --task gsm8k \
    --base-model Qwen/Qwen2.5-Coder-0.5B \
    --timeout 14400 \
    > rdagent/scenarios/rl/autorl_bench/log/rdagent.log 2>&1 &

tail -f rdagent/scenarios/rl/autorl_bench/log/rdagent.log
kill $(pgrep -f "run_agent.*rdagent")
```

## 目录结构

``` 
autorl_bench/
├── run_agent.py          # 入口：拉起 grading server + 运行 agent + 保存结果
├── conf.py               # 配置：数据/模型路径等
├── agents/               # Agent 实现
│   ├── registry.py       # Agent 注册
│   ├── example_agent/    # 示例 agent（GRPO 训练）
│   │   ├── config.yaml
│   │   ├── start.sh
│   │   └── train.py
│   └── rdagent/          # RD-Agent（自动假设+代码生成）
│       ├── config.yaml
│       └── start.sh
├── benchmark/            # 评测逻辑
│   └── benchmark.py
├── environment/          # 评测服务
│   ├── grading_server.py
│   └── instructions.txt
├── tasks/                # 任务定义
│   └── gsm8k/
│       └── description.md
├── utils/
│   └── download.py       # 模型/数据下载
├── workspace/            # [运行时生成] 工作目录
├── results/              # [运行时生成] 结果保存
└── log/                  # [运行时生成] 日志
```
