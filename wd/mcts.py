
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.utils.agent.tpl import T
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import os
from dotenv import load_dotenv
from string import Template
from pathlib import Path
import yaml
print(os.environ["CHAT_MODEL"])


from pathlib import Path
import json

json_path = Path("/data/userdata/v-lijingyuan/train_reward_stage2/comp_to_scen.json")
with json_path.open("r", encoding="utf-8") as f:
    comp_to_scen = json.load(f)

def get_response(comp_name, hypothesis_history):
    path = Path("/data/userdata/v-lijingyuan/MCTS_data/prompts_2.yaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    system_tpl = Template(data["prompts"]["hypothesis_gen"]["system"])
    system_prompt = system_tpl.substitute(
        competition_description=comp_to_scen[comp_name],
        hypothesis_history=hypothesis_history
    )

    raw = APIBackend().build_messages_and_create_chat_completion(
        user_prompt='',
        system_prompt=system_prompt,
    )


    return raw
