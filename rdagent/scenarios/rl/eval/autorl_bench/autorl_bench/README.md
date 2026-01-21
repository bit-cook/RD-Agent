# AutoRL-Bench (eval-only) — `autorl_bench/`

这里是 AutoRL-Bench 的**评测核心 Python 包**（import 名：`autorl_bench`）。它把“一个评测场景（scenario）”标准化为：

1) `scenario.yaml`（评测配置）  
2) 运行一次评测（推荐：Docker 隔离）  
3) 产出结构化结果：`status.json` + `metrics.json`（可选 `samples.jsonl`）  

> 本 README 只讲 `rdagent/scenarios/rl/eval/autorl_bench/autorl_bench/` 目录下如何使用（从 RD-Agent 仓库根目录相对路径）。上层工程说明见 `rdagent/scenarios/rl/eval/autorl_bench/README.md`。

**TL;DR（从 RD-Agent repo root 执行）**

```bash
export PYTHONPATH="$PWD/rdagent/scenarios/rl/eval/autorl_bench:${PYTHONPATH}"
python - <<'PY'
from autorl_bench.evaluator import Evaluator
print(Evaluator().run("gsm8k", overrides={"params": {"limit": 20}}))
PY
```

---

## 1. 你能用它做什么（3 种入口）

### A) Python 里直接跑（推荐）

用 `autorl_bench.evaluator.Evaluator`：

- 输入：scenario 名（例如 `gsm8k`）+ 可选 overrides
- 输出：`runs/<run_id>/` 目录（含 `status.json`、`metrics.json`、`scenario.yaml`）

### B) 起一个 HTTP 服务（FastAPI）

用 `autorl_bench.server`：

- `GET /scenarios`：列出可用场景
- `POST /runs`：创建评测（同步/异步）
- `GET /runs/{run_id}` / `.../metrics`：查询状态与指标

### C) 统一执行入口（容器内/本地）

用 `rdagent/scenarios/rl/eval/autorl_bench/env/entry.py`：

- `python env/entry.py eval --scenario <yaml> --output <dir>`
- 由 `Evaluator` 在容器中调用（默认）

---

## 2. 包结构速览

```
autorl_bench/
  benchmarks/          # 每个 benchmark 一个 Adapter（gsm8k/evalplus/miniwob）
  scenarios/           # 内置 scenario YAML（*.yaml）
  utils/               # schema / docker runner / IO / download
  evaluator.py         # 评测编排器（最常用）
  server.py            # FastAPI 服务
  __init__.py
```

---

## 3. 运行前准备

### 3.1 让 Python 能 import `autorl_bench`

从 **RD-Agent 仓库根目录**执行：

```bash
export PYTHONPATH="$PWD/rdagent/scenarios/rl/eval/autorl_bench:${PYTHONPATH}"
python -c "import autorl_bench; print('autorl_bench', autorl_bench.__version__)"
```

### 3.2 依赖与两种执行模式

- **Docker 模式（推荐）**：主机需要能访问 Docker（并安装 Python 包 `docker`；仓库里 `rdagent/utils/env.py` 会用到）。
- **本地直跑**：需要你本机安装对应 benchmark 的依赖（例如 `evalplus` / `gymnasium` / `miniwob` 等）。

上层依赖参考：`rdagent/scenarios/rl/eval/autorl_bench/requirements.txt`。

---

## 4. Quickstart：用 `Evaluator` 跑一次（Docker 推荐）

### 4.1 跑内置场景（最短可运行示例）

```bash
export PYTHONPATH="$PWD/rdagent/scenarios/rl/eval/autorl_bench:${PYTHONPATH}"

python - <<'PY'
from autorl_bench.evaluator import Evaluator

e = Evaluator()
handle = e.run("gsm8k", overrides={"params": {"limit": 20}})
print("run_id:", handle.run_id)
print("output_dir:", handle.output_dir)
print("status:", handle.status)
PY
```

### 4.2 用环境变量覆盖模型配置（常用）

这些变量会被 `env/entry.py` 读取，并透传进容器：

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="http://host.docker.internal:8000"  # openai/openai_compat 会自动补 /v1
export LLM_PROVIDER="openai_compat"
export OPENAI_MODEL="gpt-4o-mini"                          # 仅当 model_path 是 openai_compat://... 时生效
export MODEL_TEMPERATURE="0.0"
export MODEL_MAX_TOKENS="1024"
```

支持的 key：

- `OPENAI_API_KEY`
- `OPENAI_API_BASE` / `OPENAI_BASE_URL`
- `LLM_PROVIDER`
- `OPENAI_MODEL`
- `MODEL_TEMPERATURE`
- `MODEL_MAX_TOKENS`

### 4.3 覆盖 scenario（overrides 合并规则）

`Evaluator.run(..., overrides=...)` 允许覆盖 scenario 任意顶层字段，并对 `params` 与 `model` 做 dict merge：

```python
overrides = {
  "model_path": "openai_compat://my-model",
  "model": {"base_url": "http://host.docker.internal:8000/v1", "api_key": "EMPTY"},
  "params": {"limit": 50, "fewshot": 4},
}
```

---

## 5. Scenario YAML：这是最重要的接口

scenario 文件会被加载为 `autorl_bench.utils.schema.Scenario`（Pydantic）。

### 5.1 最小示例（字段解释）

```yaml
# 必填
model_path: openai_compat://my-model
data_path: hf://openai/gsm8k
baseline: baselines/gsm8k.json   # 作为 meta 记录（可为任意 YAML/JSON 友好对象）
metric: accuracy

# 可选
benchmark: gsm8k
docker_image: autorl-bench/eval-gsm8k:0.1

model:
  provider: openai_compat
  base_url: http://host.docker.internal:8000/v1
  api_key: EMPTY
  temperature: 0.0
  max_tokens: 1024

params:
  split: test
  limit: 100
```

### 5.2 scenario 如何被定位

`Evaluator` 默认会在两处找 `<scenario_name>.yaml`：

1) `autorl_bench/scenarios/`（本包内置）  
2) `../configs/scenarios/`（兼容旧位置）  

---

## 6. 输出产物：`runs/<run_id>/`

默认 `runs_dir` 是：`rdagent/scenarios/rl/eval/autorl_bench/runs/`（可通过 `Evaluator(runs_dir=...)` 改）。

一次 run 通常包含：

- `status.json`：运行状态（`running/succeeded/failed`）与时间戳，失败会带 `error`
- `scenario.yaml`：写入“合并 overrides 后的最终 scenario”（Evaluator 生成）
- `metrics.json`：结构化指标输出（统一格式）
- `samples.jsonl`：可选（轨迹/逐样本输出；如存在会在 `metrics.json.artifacts` 里登记）

`metrics.json` 结构（固定）：

```json
{
  "benchmark": "gsm8k",
  "metric": {"accuracy": 0.6247},
  "meta": {"baseline": "…", "params": {"limit": 20}},
  "artifacts": {"samples_jsonl": "samples.jsonl"}
}
```

---

## 7. Benchmarks（内置 3 个）怎么用

> 场景模板见 `autorl_bench/scenarios/*.yaml`。下面列的是各 benchmark 关心的 `data_path/params`。

### 7.1 GSM8K（`benchmarks/gsm8k_inspect.py`）

`data_path`：

- `hf://openai/gsm8k`（通过 HuggingFace datasets 拉取）
- 或本地目录：包含 `train.jsonl` / `test.jsonl`

常用 `params`：

- `split`：`train` / `test`（默认 `test`）
- `limit`：样本数上限（可选）
- `fewshot`：few-shot 数（默认 0）
- `fewshot_seed`：默认 42
- `prompt_prefix`：提示词前缀（可选）
- `answer_regex`：答案提取正则（可选；默认支持 `####`）

输出主指标：`accuracy`

### 7.2 EvalPlus（`benchmarks/evalplus_runner.py`）

`data_path`：

- `evalplus://humaneval`
- `evalplus://mbpp`

常用 `params`：

- `dataset`：`humaneval` / `mbpp`（不填则从 `data_path` 推断）
- `mode`：`two_stage`（默认）或 `auto`
- `n_samples`：生成样本数（默认 1）
- `greedy`：默认 true

说明：

- 当 `mode=two_stage` 时：会先 codegen，再 evaluate。evaluate 阶段会更“严”：`network=none`、`read_only`、`cap_drop_all` 等。

输出主指标：`pass@1`

### 7.3 MiniWoB（`benchmarks/miniwob_runner.py`）

`params`：

- `task_set`：task 列表；或一个“task 列表文件路径”（每行一个 task）
- `episodes_per_task`：默认 1
- `max_steps`：默认 30
- `seed`：默认 0
- `dom_max_elems`：默认 80（prompt 中最多展示多少 DOM 元素）

说明：

- 模型调用走 `litellm.completion`；当 `model.provider` 是 `openai/openai_compat` 且 `model_path` 没有 `openai/` 前缀时，会自动补齐。

输出主指标：`success_rate`

---

## 8. Docker 镜像：获取/构建

内置 scenarios 默认用这些镜像名：

- `autorl-bench/eval-gsm8k:0.1`
- `autorl-bench/eval-evalplus:0.1`
- `autorl-bench/eval-miniwob:0.1`

Dockerfile 位于：`rdagent/scenarios/rl/eval/autorl_bench/env/eval/`。

本地构建示例（从 RD-Agent repo root 执行）：

```bash
cd rdagent/scenarios/rl/eval/autorl_bench
docker build -f env/eval/Dockerfile.gsm8k      -t autorl-bench/eval-gsm8k:0.1      .
docker build -f env/eval/Dockerfile.evalplus   -t autorl-bench/eval-evalplus:0.1   .
docker build -f env/eval/Dockerfile.miniwob    -t autorl-bench/eval-miniwob:0.1    .
```

---

## 9. API Server：启动与调用

启动（两种方式任选其一）：

```bash
export PYTHONPATH="$PWD/rdagent/scenarios/rl/eval/autorl_bench:${PYTHONPATH}"
python -m autorl_bench.server
```

或：

```bash
uvicorn autorl_bench.server:app --host 0.0.0.0 --port 8000
```

常用调用：

```bash
curl -s http://127.0.0.1:8000/scenarios

curl -s http://127.0.0.1:8000/runs \
  -H 'content-type: application/json' \
  -d '{"scenario":"gsm8k","overrides":{"params":{"limit":20}},"sync":false}'

curl -s http://127.0.0.1:8000/runs/<run_id>
curl -s http://127.0.0.1:8000/runs/<run_id>/metrics
curl -s http://127.0.0.1:8000/runs/<run_id>/artifacts
```

---

## 10. 统一入口：`env/entry.py`（本地直跑/容器内执行）

容器内默认会跑：

```bash
python /app/env_entry.py eval --scenario /scenario.yaml --output /output
```

本地直跑（你自行保证依赖安装齐全）：

```bash
python rdagent/scenarios/rl/eval/autorl_bench/env/entry.py \
  eval \
  --scenario rdagent/scenarios/rl/eval/autorl_bench/autorl_bench/scenarios/gsm8k.yaml \
  --output /tmp/autorl_bench_out
```

---

## 11. 扩展：新增 benchmark / scenario（最短路径）

1) 新增 Adapter：在 `autorl_bench/benchmarks/` 新建文件，实现 `BenchmarkAdapter.run(...)`  
2) 注册 Adapter：编辑 `autorl_bench/benchmarks/__init__.py` 的 `_REGISTRY`  
3) 新增 scenario：在 `autorl_bench/scenarios/` 加一个 `<name>.yaml`，填好 `benchmark/docker_image/params`  
