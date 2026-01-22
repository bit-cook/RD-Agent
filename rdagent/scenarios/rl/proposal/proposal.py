import json

from rdagent.core.proposal import ExpGen, Hypothesis, Trace
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.rl.eval.task import RLTask
from rdagent.scenarios.rl.experiment.experiment import RLExperiment
from rdagent.utils.agent.tpl import T


# 默认模型路径
DEFAULT_MODEL_PATH = "/models/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_TRAIN_DATA_PATH = "/data/gsm8k/train.jsonl"



class RLPostTrainingExpGen(ExpGen):
    """RL post-training experiment generator with LLM."""

    def __init__(self, scen: Scenario | None = None):
        super().__init__(scen)

    def gen(self, trace: Trace) -> RLExperiment:
        """Generate RL post-training experiment using LLM."""
        # 构建历史摘要
        trace_summary = self._build_trace_summary(trace)

        # 调用 LLM 生成假设
        hypothesis_data = self._gen_hypothesis_with_llm(trace_summary)

        # 创建任务和实验
        rl_task = RLTask(
            name=f"RLTask_{hypothesis_data.get('algorithm', 'PPO')}",
            description=hypothesis_data.get("hypothesis", "Train RL agent"),
            model_path=DEFAULT_MODEL_PATH,
            data_path=DEFAULT_TRAIN_DATA_PATH,
        )
        hypothesis = Hypothesis(
            hypothesis=hypothesis_data.get("hypothesis", "Train RL agent"),
            reason=hypothesis_data.get("reason", ""),
            concise_reason="",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
        )
        algorithm = hypothesis_data.get("algorithm", "PPO")
        exp = RLExperiment(sub_tasks=[rl_task], hypothesis=hypothesis)
        logger.info(f"Generated experiment: {hypothesis.hypothesis} (algorithm={algorithm})")
        return exp

    def _build_trace_summary(self, trace: Trace) -> str:
        """Build summary of historical experiments."""
        if not trace or not trace.hist:
            return ""
        
        summaries = []
        for i, (exp, feedback) in enumerate(trace.hist[-3:]):  # 最近3个实验
            status = "成功" if feedback and feedback.decision else "失败"
            summaries.append(f"实验{i+1}: {exp.hypothesis.hypothesis if exp.hypothesis else 'N/A'} - {status}")
        
        return "\n".join(summaries)

    def _gen_hypothesis_with_llm(self, trace_summary: str) -> dict:
        """Generate hypothesis using LLM."""
        try:
            system_prompt = T(".prompts:hypothesis_gen.system").r()
            user_prompt = T(".prompts:hypothesis_gen.user").r(
                model_path=DEFAULT_MODEL_PATH,
                trace_summary=trace_summary,
            )
            
            resp = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            )
            return json.loads(resp)
        except Exception as e:
            # TODO: don't catch unknown exceptions
            logger.warning(f"LLM hypothesis generation failed: {e}, using default")
            return {
                "hypothesis": "Train RL agent using PPO algorithm",
                "reason": "PPO is stable and suitable for initial experiments",
                "algorithm": "PPO",
            }
