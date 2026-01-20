"""
Simple Agent - LLM decides everything

The agent has full autonomy to:
- Analyze data
- Design training strategy
- Write and execute code
- Submit when ready
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from agent.llm import LLMClient
from agent.tools import AgentTools, ToolResult, TOOL_DEFINITIONS
from agent.prompts import SYSTEM_PROMPT, STEP_RESULT_TEMPLATE, FINAL_PROMPT


class SimpleAgent:
    """
    Simple autonomous agent that uses LLM to make all decisions
    """
    
    def __init__(
        self,
        scenario_info: Dict[str, Any],
        workspace: str = None,
        max_steps: int = 50,
        llm_model: str = None,
    ):
        """
        Args:
            scenario_info: Info from /scenarios/{id} API
            workspace: Working directory for agent
            max_steps: Maximum steps before forced end
            llm_model: LLM model to use
        """
        self.scenario_info = scenario_info
        self.max_steps = max_steps
        
        # Setup workspace
        if workspace is None:
            workspace = f"./workspace/{scenario_info.get('id', 'default')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.llm = LLMClient(model=llm_model)
        self.tools = AgentTools(str(self.workspace))
        
        # State
        self.history: List[Dict[str, Any]] = []
        self.current_step = 0
        self.submitted = False
        self.final_model_path: Optional[str] = None
        
    def run(self) -> Dict[str, Any]:
        """
        Run the agent until submit or max_steps
        
        Returns:
            Dict with final_model_path and history
        """
        print(f"[Agent] Starting on scenario: {self.scenario_info.get('name')}")
        print(f"[Agent] Workspace: {self.workspace}")
        print(f"[Agent] Max steps: {self.max_steps}")
        print("-" * 50)
        
        # Build initial system prompt
        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        
        # First message to start
        messages.append({
            "role": "user",
            "content": "Please start by exploring the task. What would you like to do first?",
        })
        
        while self.current_step < self.max_steps and not self.submitted:
            self.current_step += 1
            print(f"\n[Step {self.current_step}/{self.max_steps}]")
            
            # Get LLM decision
            try:
                response = self.llm.chat(messages, json_mode=True)
                action_data = json.loads(response)
            except Exception as e:
                print(f"[Agent] LLM error: {e}")
                # Add error to history and continue
                messages.append({"role": "assistant", "content": str(e)})
                messages.append({"role": "user", "content": "Please try again with valid JSON."})
                continue
            
            # Parse action
            thought = action_data.get("thought", "")
            action = action_data.get("action", "")
            params = action_data.get("params", {})
            
            print(f"[Thought] {thought}")
            print(f"[Action] {action}")
            print(f"[Params] {params}")
            
            # Execute action
            result = self._execute_action(action, params)
            
            # Record history
            step_data = {
                "step": self.current_step,
                "thought": thought,
                "action": action,
                "params": params,
                "result": result.to_dict() if isinstance(result, ToolResult) else result,
            }
            self.history.append(step_data)
            
            # 保存每步日志到单独文件
            self._save_step_log(step_data, messages, response)
            
            # Check if submitted
            if action == "submit":
                self.submitted = True
                self.final_model_path = params.get("model_path")
                print(f"\n[Agent] Submitted model: {self.final_model_path}")
                break
            
            # Add to messages
            messages.append({"role": "assistant", "content": response})
            
            result_msg = STEP_RESULT_TEMPLATE.format(
                step=self.current_step,
                action=action,
                params=json.dumps(params, indent=2),
                result=str(result),
            )
            messages.append({"role": "user", "content": result_msg})
            
            # Truncate history if too long (keep last N messages)
            if len(messages) > 20:
                # Keep system prompt + last 18 messages
                messages = [messages[0]] + messages[-18:]
        
        # Force submit if not submitted
        if not self.submitted:
            print(f"\n[Agent] Max steps reached, forcing submit...")
            self.final_model_path = self._force_submit(messages)
        
        # Save history
        self._save_history()
        
        return {
            "final_model_path": self.final_model_path,
            "total_steps": self.current_step,
            "submitted": self.submitted,
            "workspace": str(self.workspace),
        }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with scenario info"""
        return SYSTEM_PROMPT.format(
            scenario_name=self.scenario_info.get("name", "Unknown"),
            scenario_description=self.scenario_info.get("description", ""),
            base_model_path=self.scenario_info.get("base_model_path", ""),
            train_data_path=self.scenario_info.get("train_data_path", ""),
            test_data_path=self.scenario_info.get("test_data_path", ""),
            baseline_score=self.scenario_info.get("baseline_score", 0),
            metric=self.scenario_info.get("metric", "accuracy"),
            max_steps=self.max_steps,
            current_step=self.current_step,
            workspace=str(self.workspace),
        )
    
    def _execute_action(self, action: str, params: Dict[str, Any]) -> ToolResult:
        """Execute an action with given parameters"""
        
        if action == "run_code":
            code = params.get("code", "")
            return self.tools.run_code(code)
            
        elif action == "read_file":
            path = params.get("path", "")
            return self.tools.read_file(path)
            
        elif action == "write_file":
            path = params.get("path", "")
            content = params.get("content", "")
            return self.tools.write_file(path, content)
            
        elif action == "list_dir":
            path = params.get("path", ".")
            return self.tools.list_dir(path)
            
        elif action == "submit":
            model_path = params.get("model_path", "")
            return ToolResult(
                success=True,
                output=f"Submitting model: {model_path}",
            )
            
        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown action: {action}",
            )
    
    def _force_submit(self, messages: List[Dict]) -> Optional[str]:
        """Force agent to submit when max steps reached"""
        messages.append({
            "role": "user",
            "content": FINAL_PROMPT.format(max_steps=self.max_steps),
        })
        
        try:
            response = self.llm.chat(messages, json_mode=True)
            data = json.loads(response)
            return data.get("params", {}).get("model_path")
        except:
            # Return last saved model if any
            return None
    
    def _save_step_log(self, step_data: Dict, messages: List[Dict], response: str):
        """保存每步详细日志"""
        step_file = self.workspace / f"step_{step_data['step']:03d}.json"
        
        log_data = {
            "step": step_data["step"],
            "timestamp": datetime.now().isoformat(),
            
            # LLM 输入
            "llm_input": {
                "messages_count": len(messages),
                "last_user_message": messages[-1]["content"][:500] if messages else "",
            },
            
            # LLM 输出 (解析后)
            "llm_output": {
                "thought": step_data["thought"],
                "action": step_data["action"],
                "params": step_data["params"],
            },
            
            # 执行结果
            "result": step_data["result"],
            
            # 完整 LLM 响应 (原始)
            "raw_response": response,
        }
        
        with open(step_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"[Log] Step {step_data['step']} -> {step_file.name}")
    
    def _save_history(self):
        """Save run history to file"""
        history_file = self.workspace / "summary.json"
        data = {
            "scenario": self.scenario_info.get("id"),
            "max_steps": self.max_steps,
            "total_steps": self.current_step,
            "submitted": self.submitted,
            "final_model_path": self.final_model_path,
            "history": self.history,
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Agent] Summary saved to: {history_file}")


def run_agent(
    scenario_id: str = "gsm8k",
    max_steps: int = 50,
    api_base: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """
    Convenience function to run agent on a scenario
    
    Args:
        scenario_id: Scenario ID
        max_steps: Maximum steps
        api_base: AutoRL-Bench API base URL
        
    Returns:
        Agent results
    """
    import requests
    
    # Get scenario info
    print(f"[Agent] Getting scenario info: {scenario_id}")
    resp = requests.get(f"{api_base}/scenarios/{scenario_id}")
    resp.raise_for_status()
    scenario_info = resp.json()
    
    # Run agent
    agent = SimpleAgent(scenario_info, max_steps=max_steps)
    result = agent.run()
    
    # Submit for evaluation if we have a model
    if result.get("final_model_path"):
        print(f"\n[Agent] Evaluating model...")
        eval_resp = requests.post(
            f"{api_base}/evaluate",
            json={
                "scenario_id": scenario_id,
                "model_path": result["final_model_path"],
            }
        )
        if eval_resp.ok:
            eval_result = eval_resp.json()
            result["evaluation"] = eval_result
            print(f"[Agent] Score: {eval_result.get('score')}%")
            print(f"[Agent] Improvement: {eval_result.get('improvement')}%")
    
    return result


if __name__ == "__main__":
    # Test run
    result = run_agent(scenario_id="gsm8k", max_steps=10)
    print("\n" + "=" * 50)
    print("Final Result:")
    print(json.dumps(result, indent=2))

