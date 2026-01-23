
# 任务描述

我们正在开发一个最简版的 RL 后训练基准测试。开发时遵循以下原则：
- 保持代码简洁是最高优先级
- 性能不在考虑范围内

## 技术决策：

- 我们不想重新发明仓库级代码生成。所以打算使用现有的 coder 来生成仓库级代码。
  - 候选：aider, openhands

## TODO:

- [ ] (xiao)repo-level coder may not provide interfaces that fits curernt CoSTEER's interface.
  - related code:
    - `rdagent/components/coder/CoSTEER/evolving_strategy.py`

# 编码原则
实现新代码时不要捕获未知异常。我倾向于让错误传播，以便及时发现和修复。

# 测试原则
log 都输出在这并且同一功能复用同一log命名：program/RD-Agent/rdagent/scenarios/rl/eval/autorl_bench/log

# 潜在重构待办
## 框架
- 简化构建新 CoSTEER coder 的流程 (xiao 正在思考)
  - 相关代码: `rdagent/components/coder/rl/costeer.py`
- 在 `rdagent/core/experiment.py` 中：能否在 Generic 类中创建新的 Workspace？
