# AutoRL-Bench

LLM Post-Training 强化学习评测基准（bench）。  
注意：AutoRL-Bench 是**独立的 benchmark**，RD-Agent 是后续要优化的 agent。本阶段以 benchmark 为主。

## 项目架构

AutoRL-Bench 由三层构成：

1. **任务与评测层**（`tasks/`, `environment/`）  
   定义任务描述、数据格式与评测接口（Grading Server）。
2. **运行与编排层**（`run_agent.py`, `conf.py`）  
   负责拉起 grading server、准备 workspace、运行 agent、收集结果。
3. **Agent 层**（`agents/`）  
   具体训练/推理策略的实现，可替换、可扩展。

## 运行流程

1. **准备资源**：检查/下载模型与数据（`models/`, `datasets/`）。
2. **初始化工作区**：创建 `workspace/<task>/`，并建立数据与模型的软链接。
3. **启动评测服务**：起 grading server 提供 `/submit` `/best` 等接口。
4. **运行 agent**：执行 agent 的入口脚本（如 `agents/example_agent/`）。
5. **收集结果**：保存 `results/`，包含 best score、日志、模型链接等。

## 快速开始

```bash
# 进入 AutoRL-Bench 目录（独立运行 benchmark）
cd /Data/home/v-wanyichen/cwy/program/RD-Agent/rdagent/scenarios/rl/autorl_bench

# 后台运行
nohup python run_agent.py \
    --agent-id example_agent \
    --task gsm8k \
    --base-model Qwen/Qwen2.5-0.5B \
    --timeout 7200 \
    > log/run.log 2>&1 &

# 查看日志
tail -f log/run.log

# 停止
kill $(pgrep -f "run_agent.py.*example_agent")
```

## 目录结构

``` 
autorl_bench/
├── run_agent.py          # 入口：拉起 grading server + 运行 agent + 保存结果
├── conf.py               # 配置：数据/模型/workspace 目录等
├── benchmark/            # 评测逻辑封装
│   └── benchmark.py      # run_benchmark 主入口
├── agents/               # Agent 实现与注册
│   ├── registry.py       # Agent registry
│   └── example_agent/    # 示例 agent
│       ├── start.sh      # 入口脚本（由 run_agent 调用）
│       ├── train.py      # 训练逻辑示例
│       └── config.yaml   # 示例配置
├── environment/          # 评测环境（grading server）
│   ├── grading_server.py # 评分服务
│   └── instructions.txt  # 评测说明
├── tasks/                # 任务定义
│   ├── __init__.py       # 任务注册与元信息
│   └── gsm8k/            # 示例任务
│       └── description.md
├── utils/                # 下载与通用工具
│   └── download.py
├── workspace/            # 运行时工作目录（每个 task 一个 workspace）
├── results/              # 运行结果（best score / logs / 模型软链）
├── log/                  # 运行日志（如 run.log）
├── doc/                  # 研究/笔记文档
└── test/                 # 本地测试代码
```

## 环境变量

```bash
export AUTORL_FILE_PATH=/path/to/data
```
