# AutoRL-Bench 架构设计

> 评测 Agent 自动进行 RL Post-training 的能力

---

## 1. 项目定位

**我们评测的是 Agent，不是模型。**

- 传统 Benchmark：评测模型能力
- AutoRL-Bench：评测 "让模型变好的 Agent" 的能力

## 2. 项目结构

```
AutoRL-Bench/
├── autorl_bench/           # Benchmark 核心
│   ├── server.py           # API 服务
│   ├── scenarios/          # 场景定义
│   │   ├── base.py
│   │   └── gsm8k.py
│   └── evaluator.py        # 评测器
|---env/
|     entry.py
|     training/
|        Dockerfile
|     eval/
|        Dockerfile
├── assets/                 # 资源 (不上传 git)
│   ├── data/
│   │   ├── gsm8k/          # 数学数据集
│   │   └── humaneval/      # 代码数据集
│   └── models/
│       └── Qwen2.5-3B-Instruct/
│
│
├── log/                    # 服务日志
│   ├── server.log
│   └── agent.log
│
├── configs/                # 配置文件
└── test/                   # 测试脚本
```

### Intefaces

#### CLI interfaces
- TODO:
  - Interfaces:
    - e.g. `python -m rdagent.scenario.rl.eval.<scenario-name>.server`
    - ... download data ...

- Who will implement the Agent code
  - (We choose this)**RD-Agent is reponsible for the agent code**
    - e.g. `python -m rdagent.scenario.rl.env.entry  --type "eval"  --workspace ..`.
      - there may be a implied entry in the workspace like `<workspace>/agent.py`
  - benchmark will provide the agent code
    - `python -m rdagent.scenario.rl.env.entry  --type "eval"  --workspace ..`.
      - It will mount agent code into the workspace.
      - benchmark's agent code will load a model in a specific folder like `<workspace>/model`

- Training model:
  - `python -m rdagent.scenario.rl.env.entry  --type "training"  --workspace ..`.
    - the workspace will contain a entry named "train.py"

#### RESTful interface



#### Filesystem interface


```

├── workspace/              # Agent 工作区 (不上传 git)
│   └── gsm8k_YYYYMMDD_HHMMSS/
│       ├── logs/           # 给人看的日志
│       │   ├── step_001.json
│       │   ├── step_002.json
│       │   └── summary.json
│       └── work/           # Agent 的工作目录
│           ├── train.py
│           └── output/     # 训练后的模型

├    ── agent/                  # Demo Agent
│       ├── simple_agent.py     # LLM 全自主 Agent
│       ├── tools.py            # Agent 可用工具
│       ├── llm.py              # LLM 客户端 (重试机制)
│       └── prompts.py          # System Prompt
│
```

## 3. API 设计

**只有 2 个核心接口：**

### GET /scenarios/{id}

获取任务信息，Agent 根据这些信息自己做 RL 训练。

```json
// GET /scenarios/gsm8k
{
  "id": "gsm8k",
  "name": "GSM8K Math Reasoning",
  "description": "小学数学应用题",
  
  "base_model_path": "assets/models/Qwen2.5-3B-Instruct",
  "train_data_path": "assets/data/gsm8k/train.jsonl",
  "test_data_path": "assets/data/gsm8k/test.jsonl",
  
  "baseline_score": 62.47,
  "metric": "accuracy"
}
```

### POST /evaluate

Agent 训练完模型后，提交评测。

```json
// Request
{
  "scenario_id": "gsm8k",
  "model_path": "/path/to/trained_model"
}

// Response
{
  "score": 75.2,
  "baseline": 62.47,
  "improvement": 12.73,
  "details": {
    "correct": 992,
    "total": 1319
  }
}
```

## 4. 使用流程

```
Agent                              AutoRL-Bench
  │                                     │
  │  1. GET /scenarios/gsm8k           │
  │────────────────────────────────────►│
  │◄────────────────────────────────────│
  │     {data_path, model_path, ...}    │
  │                                     │
  │  2. Agent 自己做 RL 训练            │
  │  ┌──────────────────────────────┐   │
  │  │ - 读数据                     │   │
  │  │ - 加载模型                   │   │
  │  │ - 选框架 (trl/verl/自己写)   │   │
  │  │ - 设计 reward               │   │
  │  │ - 训练、保存模型            │   │
  │  └──────────────────────────────┘   │
  │                                     │
  │  3. POST /evaluate                  │
  │────────────────────────────────────►│
  │◄────────────────────────────────────│
  │     {score: 75.2, improvement: ...} │
```

## 5. 核心原则

| 原则 | 说明 |
|------|------|
| **Agent 黑盒** | 不限制框架、算法、训练方式 |
| **接口极简** | 只有 2 个接口 |
| **本地文件** | 数据/模型预下载，API 返回路径 |
| **只评测结果** | 不关心过程，只看最终分数 |

## 6. 当前状态

### GSM8K 场景

| 项目 | 值 |
|------|-----|
| 基础模型 | Qwen2.5-3B-Instruct |
| 训练集 | 7,473 条 |
| 测试集 | 1,319 条 |
| Baseline | **62.47%** |
| 评测指标 | Accuracy |

### 已完成

- [x] GSM8K 数据集下载
- [x] HumanEval 数据集下载
- [x] Qwen2.5-3B-Instruct 模型下载
- [x] Baseline 评测 (62.47%)
- [x] 评测结果保存 (results/)
- [x] Simple Agent 实现
- [x] Agent 日志系统 (step_xxx.json, summary.json)
- [x] LLM 客户端 (重试机制)

### 待完成

- [ ] 工具权限限制 (防止 Agent 删除关键文件)
- [ ] workspace 目录分离 (logs/ + work/)
- [ ] HumanEval 场景实现 (代码评测)
- [ ] Docker 隔离 (可选)

---

## 7. Demo Agent 设计

**最简设计：LLM 全自主决策**

```
┌──────────────────────────────────────────────────────────────┐
│                     Simple Agent                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  初始化:                                                      │
│  - 场景信息 (model_path, data_path, baseline)                │
│  - 可用工具列表                                               │
│  - 最大步数 max_steps                                        │
│                                                              │
│  while step < max_steps and not submitted:                   │
│      LLM 决定下一步:                                          │
│      - run_code(code)      执行代码                          │
│      - read_file(path)     读文件                            │
│      - submit(model_path)  提交 → 结束                       │
│                                                              │
│  超时 → 强制结束                                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Agent 核心代码

```python
class SimpleAgent:
    def __init__(self, scenario_info: dict, max_steps: int = 50):
        self.info = scenario_info
        self.max_steps = max_steps
        self.history = []
    
    def run(self) -> dict:
        for step in range(self.max_steps):
            # LLM 根据历史决定下一步
            action = self.llm_decide(self.history)
            
            if action.type == "submit":
                return self.evaluate(action.model_path)
            
            result = self.execute(action)
            self.history.append((action, result))
        
        # 超时强制结束
        return self.force_submit()
```

### System Prompt

```
你是 RL 训练专家。任务：让小模型在 GSM8K 上取得更好成绩。

## 场景
- 模型: {model_path}
- 数据: {train_data_path}
- Baseline: {baseline_score}%

## 工具
1. run_code(code) - 执行 Python 代码
2. read_file(path) - 读取文件
3. submit(model_path) - 提交评测，结束任务

## 规则
- 最多 {max_steps} 步，当前第 {step} 步
- 可随时 submit 提前结束
- 超时强制结束
```

### 设计原则

| 原则 | 说明 |
|------|------|
| **LLM 全自主** | 不限制流程，LLM 自己决定做什么 |
| **工具极简** | 只给 run_code + read_file + submit |
| **步数限制** | 防止无限循环，可提前提交 |
| **历史透明** | 每步结果都反馈给 LLM |

---

## 附录：Baseline 结果

```
模型: Qwen2.5-3B-Instruct
数据: GSM8K test (1319 条)
准确率: 62.47% (824/1319)
结果文件: results/gsm8k/baseline_Qwen2.5-3B-Instruct_20260118_111008.json
```
