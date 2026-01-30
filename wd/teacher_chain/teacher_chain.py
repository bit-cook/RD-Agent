import json
from collections import defaultdict
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
load_dotenv("/data/userdata/v-lijingyuan/RD-Agent-fix-mcts/RD-Agent/.env")
from rdagent.utils.agent.tpl import T
from rdagent.oai.llm_utils import APIBackend, md5_hash
from typing import Any, Dict, List, Optional, Tuple
import numpy as np 

import torch
import concurrent.futures
import json
import numpy as np
import torch.multiprocessing as mp

import os
import json
import time
import traceback
from multiprocessing import Process, Queue, set_start_method
from typing import Any, Dict


def build_teacher_chain_topk(data, k=2):
    groups = defaultdict(list)
    for item in data:
        inp = item["input"]
        key = (inp["exp_name"], inp["comptation_name"])
        groups[key].append(inp)

    teacher_chain = {}

    for key, items in groups.items():
        bigger_is_better = items[0]["bigger_is_better"]

        if bigger_is_better == 1:
            sorted_items = sorted(items, key=lambda x: x["valid_score"], reverse=True)
        else:
            sorted_items = sorted(items, key=lambda x: x["valid_score"])

        topk = sorted_items[:k]

        teacher_chain[key] = [
            {
                "comptation_name": x["comptation_name"],
                "hypothesis_chain": x["hypothesis_chain"],
                "score": x["valid_score"]
            }
            for x in topk
        ]
    return teacher_chain



class RewardModelInference(nn.Module):
    def __init__(self, base_model_name, adapter_path, reward_head_path, device="cuda"):
        super().__init__()
        self.device = device
        self.base = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
        self.base = PeftModel.from_pretrained(self.base, adapter_path).to(device)
        if hasattr(self.base, "gradient_checkpointing_enable"):
            self.base.gradient_checkpointing_enable()
        if hasattr(self.base.config, "use_cache"):
            self.base.config.use_cache = False
        hs = getattr(self.base.config, "hidden_size",
                     getattr(self.base.config, "n_embd",
                     getattr(self.base.config, "d_model", None)))
        if hs is None:
            hs = self.base.get_input_embeddings().embedding_dim

        self.reward_head = nn.Linear(hs, 1).to(device)
        self.reward_head.load_state_dict(torch.load(reward_head_path, map_location=device))

    @staticmethod
    def pool_last_nonpad(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        lengths = attn_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
        return last_hidden.gather(1, idx).squeeze(1)

    def forward(self, input_ids, attention_mask):
        out = self.base(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
            use_cache=False
        )
        last_hidden = out.hidden_states[-1]
        pooled = self.pool_last_nonpad(last_hidden, attention_mask)
        reward = self.reward_head(pooled).squeeze(-1)
        return reward

    def compute_reward(self, texts, tokenizer,comp_description, system_prompt=None):
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif not hasattr(self, "system_prompt"):
            self.system_prompt = (
                "You are a senior data science competition judge and solution expert.\n"
                "Your task is to evaluate the quality, reasoning progression, and innovation of hypothesis chains.\n"
                "A hypothesis chain shows iterative improvement of solutions.\n"
                "You should assess:\n"
                "1) reasoning correctness and consistency across steps,\n"
                "2) improvement and refinement through the chain,\n"
                "3) final hypothesis quality and practicality.\n"
                "Be strict and fair. Provide expert-level insight."
            )

        inputs = []
        for s in texts:
            prompt = (
                f"{self.system_prompt}\n\n"
                f"Competition description:\n{comp_description}\n\n"
                "Hypothesis Chain (each step separated by '->'):\n"
                f"{s}\n\n"
                "<think>\n"
                "Analyze the evolution of hypotheses, step-by-step, identifying strengths, weaknesses, and logical progression.\n"
                "Focus on clarity, correctness, and improvement.\n"
                "Make sure to consider the chain direction from earliest to latest.\n"
                "</think>\n\n"
                "Final Evaluation:\n"
            )

            inputs.append(prompt)

        enc = tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=2300,
            return_tensors="pt"
        )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        rewards = self.forward(enc["input_ids"], enc["attention_mask"])

        return torch.exp(rewards).cpu().tolist()
    

prompt_template = """
You are a data science expert and Kaggle Grandmaster.

Below is a competition scenario description:
{SCENARIO}

Below is an optimization chain derived from iterative feedback:
{CHAIN}

Notes:
- The optimization chain is composed of multiple reasoning steps.
- Each step is separated by the delimiter "->".
- A step may be a diagnosis, feedback, or hypothesis derived from iterative refinement.

Your tasks:
1. Identify one reasoning step that is overly specific, tied to a particular dataset, or clearly looks like a feedback-proposed hypothesis, and remove it.
2. insert a new, generalized hypothesis that is logically consistent with the competition scenario in place of the removed step.
3. Maintain the original "->" delimiter structure.
4. Ensure the rewritten chain is coherent, logically sound, and preserves the intended optimization flow.
5. Do NOT remove all steps; keep the chain meaningful.
6. **Generalization requirement**: When inserting hypotheses, abstract them so they are broadly applicable to a wide range of competitions. Avoid overly specific references to a particular dataset, feature, or model unless absolutely necessary.
7. Focus on creating hypotheses that capture general strategies, principles, or approaches in data science competitions, making them useful for multiple scenarios.

Output:
- A single rewritten optimization chain using "->" as the delimiter.
""".strip()



evo = [ "cassava-leaf-disease-classification",
            "h-and-m-personalized-fashion-recommendations",
            "jigsaw-toxic-comment-classification-challenge",
            "leaf-classification",
            "tweet-sentiment-extraction",
            "us-patent-phrase-to-phrase-matching",
            "whale-categorization-playground",
            "learning-agency-lab-automated-essay-scoring-2",
            "aptos2019-blindness-detection",
            "kuzushiji-recognition",
            "herbarium-2020-fgvc7",
            "text-normalization-challenge-russian-language",
            "rsna-miccai-brain-tumor-radiogenomic-classification",
            "freesound-audio-tagging-2019",
            "mlsp-2013-birds",
            "spooky-author-identification",
            "hubmap-kidney-segmentation",]



from transformers import AutoTokenizer
import torch
import os

logdir = "/data/userdata/v-lijingyuan/last_run_5"
base_model = "Qwen/Qwen3-0.6B"
adapter_path = os.path.join(logdir, "lora_adapter")
reward_head_path = os.path.join(logdir, "reward_head.pt")

tokenizer = AutoTokenizer.from_pretrained(base_model)
if not getattr(tokenizer, "pad_token", None):
    tokenizer.pad_token = tokenizer.eos_token

model = RewardModelInference(
    base_model_name=base_model,
    adapter_path=adapter_path,
    reward_head_path=reward_head_path,
)

# --- 多卡推理 ---
# model = torch.nn.DataParallel(model)   # wrap
# model = model.cuda()                   # move to all GPUs automatically
# model.eval()


final_teacher_chain = []
MAX_TRIALS = 5
THRESHOLD = 0.1
NUM_GPUS = 4

# 分配GPU
gpu_ids = list(range(NUM_GPUS))
models = []

for gpu_id in gpu_ids:
    model_gpu = RewardModelInference(
        base_model_name=base_model,
        adapter_path=adapter_path,
        reward_head_path=reward_head_path,
        # device_map=None 确保直接加载到指定 GPU
    ).to(f"cuda:{gpu_id}")
    model_gpu.eval()
    models.append(model_gpu)

# --- 处理单个样本 ---
def process_item_on_gpu(item, gpu_id):
    import torch
    with open("/data/userdata/v-lijingyuan/dpo/comp_to_scen.json", "r", encoding="utf-8") as f:
        comp_to_scen = json.load(f)
    # 每个进程独立创建 APIBackend
    api = APIBackend()

    # 每个进程自己初始化 GPU 模型副本
    model_gpu = RewardModelInference(
        base_model_name=base_model,
        adapter_path=adapter_path,
        reward_head_path=reward_head_path,
        device= f"cuda:{gpu_id}"
    ).to(f"cuda:{gpu_id}")
    model_gpu.eval()

    comp = item["comptation_name"]
    chain = item["hypothesis_chain"]
    if comp not in evo:
        return None

    scenario = comp_to_scen[comp]

    with torch.no_grad():
        base_score = model_gpu.compute_reward([chain], tokenizer, scenario)

    current_chain = chain
    improved_chain = None
    for _ in range(MAX_TRIALS):
        prompt = prompt_template.format(SCENARIO=scenario, CHAIN=current_chain)
        new_chain = api.build_messages_and_create_chat_completion(
            system_prompt=prompt,
            user_prompt=''
        )
        with torch.no_grad():
            new_score = model_gpu.compute_reward([new_chain], tokenizer, scenario)

        if np.abs(new_score[0] - base_score[0]) < THRESHOLD:
            improved_chain = new_chain
            break
        current_chain = new_chain

    return {
        "comptation_name": comp,
        "hypothesis_chain": chain,
        "new_hypothesis_chain": improved_chain if improved_chain else current_chain
    }



#mp.set_start_method("spawn", force=True)

def process_all(flat_list):
    save_path = "/data/userdata/v-lijingyuan/rl_pipe_line/final_teacher_chain_v1.jsonl"
    # 清空之前的文件
    open(save_path, "w", encoding="utf-8").close()

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = [executor.submit(process_item_on_gpu, item, i % NUM_GPUS)
                   for i, item in enumerate(flat_list)]

        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res:
                    # 实时写入，每行一个 JSON 对象
                    with open(save_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
            except Exception as e:
                print("Error processing item:", e)

# --- 执行 ---
# final_teacher_chain = process_all(flat_list)

# # --- 保存结果 ---
# save_path = "final_teacher_chain.json"
# with open(save_path, "w", encoding="utf-8") as f:
#     json.dump(final_teacher_chain, f, ensure_ascii=False, indent=4)
# print(f"Saved to {save_path}")


# AR  text ->   token    1  -> 1
# AR  NF VQ -> token - >  3
# AR  DDPM VQ -> token - > 2
# AR  FLOW match ->token - > 5
# AR  token -> token - >  4

if __name__ == "__main__":
    # 必须在 __main__ 下启动多进程
    with open("/data/userdata/v-lijingyuan/dpo/comp_to_scen.json", "r", encoding="utf-8") as f:
        comp_to_scen = json.load(f)
    with open("/data/userdata/v-lijingyuan/dpo/final_pairs_diff_1.json", "r") as f:
        data = json.load(f)

    teacher = build_teacher_chain_topk(data, k=3)
    flat_list = []
    for key, items in teacher.items():
        flat_list.extend(items)

    with open("teacher_chain_top3.json", "w", encoding="utf-8") as f:
        json.dump(flat_list, f, ensure_ascii=False, indent=2)


    flat_list = flat_list
    mp.set_start_method("spawn", force=True)

    final_teacher_chain = process_all(flat_list)

    # save_path = "final_teacher_chain.json"
    # with open(save_path, "w", encoding="utf-8") as f:
    #     json.dump(final_teacher_chain, f, ensure_ascii=False, indent=4)

    #print(f"Saved to {save_path}")