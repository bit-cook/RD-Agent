# AutoRL-Bench

> 评测 Agent 自动进行 RL Post-training 的能力

## 简介

AutoRL-Bench 是一个评测 Benchmark，用于评估 Agent 自动进行 RL（强化学习）后训练的能力。

**我们提供：**
- 标准化的 API 接口
- 预定义的场景（如 GSM8K 数学题）
- 统一的评估方法

**Agent 需要做：**
- 调用 API 获取任务信息
- 自己选择算法、框架进行训练
- 提交模型进行评估

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 或者直接安装包
pip install -e .
```

## 快速开始

### 1. 下载数据和模型

```bash
# 下载 GSM8K 数据集和基础模型
python -m autorl_bench.utils.download --scenario gsm8k
```

### 2. 启动 API 服务

```bash
# 启动服务器
python -m autorl_bench.server
# 服务运行在 http://localhost:8000
```

### 3. 使用 API

```python
import requests

# 获取场景列表
resp = requests.get("http://localhost:8000/scenarios")
print(resp.json())

# 获取 GSM8K 场景详情
resp = requests.get("http://localhost:8000/scenarios/gsm8k")
info = resp.json()
print(f"训练数据: {info['train_data']}")
print(f"基础模型: {info['base_model_path']}")
print(f"基线分数: {info['baseline_score']}")

# Agent 自己训练模型...
# (使用 verl, TRL, 或其他框架)

# 评估模型
resp = requests.post(
    "http://localhost:8000/evaluate",
    json={
        "scenario_id": "gsm8k",
        "model_path": "/path/to/your/trained/model"
    }
)
result = resp.json()
print(f"分数: {result['score']}")
print(f"提升: {result['improvement']}")
```

## API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/scenarios` | GET | 获取所有场景列表 |
| `/scenarios/{id}` | GET | 获取场景详情 |
| `/interact` | POST | 环境交互（在线场景） |
| `/evaluate` | POST | 评估训练好的模型 |

## 目录结构

```
AutoRL-Bench/
├── autorl_bench/
│   ├── __init__.py
│   ├── server.py           # API 服务
│   ├── scenarios/          # 场景定义
│   │   ├── base.py         # 基类
│   │   └── gsm8k.py        # GSM8K 场景
│   ├── evaluator.py        # 评估器
│   └── utils/
│       └── download.py     # 下载工具
├── configs/
│   └── scenarios/
│       └── gsm8k.yaml      # GSM8K 配置
├── doc/
│   └── architecture.md     # 架构文档
├── requirements.txt
└── README.md
```

## 支持的场景

| 场景 | 类型 | 描述 |
|------|------|------|
| GSM8K | offline | 小学数学题，On-policy RL |

更多场景开发中...

## 设计原则

1. **接口简单** - 只有 4 个 API
2. **Agent 黑盒** - 不限制 Agent 实现方式
3. **评估标准化** - 统一的评估流程

## License

MIT
