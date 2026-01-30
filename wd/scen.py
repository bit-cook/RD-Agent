import json
from pathlib import Path
from typing import Dict
from rdagent.utils.agent.tpl import T
from rdagent.scenarios.data_science.scen.utils import describe_data_folder_v2
from rdagent.oai.llm_utils import APIBackend


class DataScienceScen:
    """Simplified Data Science Scenario for generating full scenario description."""

    def __init__(self, competition: str, local_data_path: str):
        self.competition = competition
        self.local_data_path = local_data_path

        # 1) 获取比赛描述
        self.raw_description = self._get_description()
        # 2) 获取数据文件夹描述
        self.processed_data_folder_description = self._get_data_folder_description()
        # 3) 分析描述生成各类信息
        self._analysis_competition_description()
        # 方向信息
        self.metric_direction: bool = getattr(self, "metric_direction_guess", True)

    def _get_description(self):
        """Load description.md or competition.json"""
        desc_md = Path(self.local_data_path) / self.competition / "description.md"
        desc_json = Path(self.local_data_path) / f"{self.competition}.json"
        if desc_md.exists():
            return desc_md.read_text()
        elif desc_json.exists():
            with desc_json.open("r") as f:
                return json.load(f)
        else:
            return f"No description found for {self.competition}."

    def _get_data_folder_description(self) -> str:
        """Describe the data folder."""
        return describe_data_folder_v2(Path(self.local_data_path) / self.competition, show_nan_columns=False)

    def _analysis_competition_description(self):
        """Analyze description using LLM to extract structured info."""
        sys_prompt = T(".prompts:competition_description_template.system").r()
        user_prompt = T(".prompts:competition_description_template.user").r(
            competition_raw_description=self.raw_description,
            competition_processed_data_folder_description=self.processed_data_folder_description,
        )
        response_analysis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | int | bool],
        )
        response_json_analysis = json.loads(response_analysis)

        self.task_type = response_json_analysis.get("Task Type", "No type provided")
        self.data_type = response_json_analysis.get("Data Type", "No data type provided")
        self.brief_description = response_json_analysis.get("Brief Description", "")
        self.dataset_description = response_json_analysis.get("Dataset Description", "")
        self.submission_specifications = response_json_analysis.get("Submission Specifications", "")
        self.metric_description = response_json_analysis.get("Metric Evaluation Description", "")
        self.metric_name = response_json_analysis.get("Metric Name", "custom_metric")
        self.metric_direction_guess = response_json_analysis.get("Metric Direction", True)

    @property
    def background(self) -> str:
        """Return background prompt."""
        return T(".prompts:competition_background").r(
            task_type=self.task_type,
            data_type=self.data_type,
            brief_description=self.brief_description,
            dataset_description=self.dataset_description,
            model_output_channel=1,
            metric_description=self.metric_description,
        )

    def get_scenario_all_desc(self, eda_output=None) -> str:
        """Return full scenario description."""
        return T(".prompts:scenario_description").r(
            background=self.background,
            submission_specifications=self.submission_specifications,
            evaluation=self.metric_description,
            metric_name=self.metric_name,
            metric_direction=self.metric_direction,
            raw_description=self.raw_description,
            use_raw_description=True,
            time_limit="1.00 hours",
            recommend_time_limit="1.00 hours",
            eda_output=eda_output,
            debug_time_limit="1.00 minutes",
            recommend_debug_time_limit="1.00 minutes",
            runtime_environment="Minimal runtime environment",
        )


# ======= 测试示例 =======
if __name__ == "__main__":
    competition_name = "aerial-cactus-identification"
    local_data_path = "/data/userdata/share/mle_kaggle"

    scenario = DataScienceScen(competition_name, local_data_path)
    desc = scenario.get_scenario_all_desc(eda_output="EDA summary here")
    print(desc)
