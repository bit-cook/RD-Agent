
# Task Description

We are developing a most naive version of RL post-training benchmark. When developing the benchmark, we are following principles below:
- Keeping code simple is the highest priority.
- Performance is not in our consideration.

## Technical decisions:

- We don't want to re-invent the repo-level coding. So we want to employ exsiting coder to generate repository-level code.
  - candidates: aider, openhands.

## TODO:

- [ ] (xiao)repo-level coder may not provide interfaces that fits curernt CoSTEER's interface.
  - related code:
    - `rdagent/components/coder/CoSTEER/evolving_strategy.py`

# Coding Principles
Don't catch unknown exceptions when implementing new code. I prefer to let the error propagate so it can be detected and fixed promptly.

# Potential Refactoring backlog
## Framework
- Make it simpler to build a new CoSTEER coder (xiao is thinking about it).
  - related code: `rdagent/components/coder/rl/costeer.py`
-  in `rdagent/core/experiment.py`:  can we create a new Workspace in the Generic class?
