import os
import json
import math
from dataclasses import dataclass
from collections import defaultdict
from multiprocessing import Process, Lock

from mcts import get_response
from reward_inference import RewardModelInference
from transformers import AutoTokenizer
import multiprocessing as mp
from multiprocessing import Pool, set_start_method


# ==================  全局输出路径  ==================
BASE_DIR = "/data/userdata/v-lijingyuan/MCTS_data/mcts_data_new"
os.makedirs(BASE_DIR, exist_ok=True)

PAIR_LOG = os.path.join(BASE_DIR, "all_pairs_gpt5.jsonl")
TRAJ_LOG = os.path.join(BASE_DIR, "all_trajs_gpt5.jsonl")


lock = Lock()   # 🔒 进程写入锁


# ===================================================
# 工具：安全写入 jsonl
# ===================================================
def append_jsonl(path, rows):
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ===================================================
# 数据结构
# ===================================================
@dataclass
class Trajectory:
    state: list[str]
    parent: list[str]
    world_value: float
    visits: int


def traj_to_dict(t: Trajectory):
    return {
        "state": t.state,
        "parent": t.parent,
        "state_str": "->".join(t.state),
        "world_value": t.world_value,
        "visits": t.visits,
        "length": len(t.state),
    }


# ===================================================
# MCTS 节点
# ===================================================
class MCTSNode:
    def __init__(self, state=None, parent=None):
        self.state = state or []
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.world_values = 0.0
        self.ps = 0.0

    def best_child(self, c=1.41):
        best_score = -float("inf")
        best_child = None
        for child in self.children.values():
            if child.visits == 0:
                uct = float("inf")
            else:
                uct = child.value / child.visits + c * math.sqrt(
                    math.log(self.visits) / child.visits
                )
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child


# ===================================================
# 收集轨迹
# ===================================================
def collect_trajectories(root: MCTSNode):
    trajs = []

    def dfs(node):
        if not node.children:  # leaf
            trajs.append(
                Trajectory(
                    state=node.state.copy(),
                    parent=node.parent.state.copy() if node.parent else [],
                    world_value=float(node.world_values),
                    visits=node.visits,
                )
            )
        for ch in node.children.values():
            dfs(ch)

    dfs(root)
    return trajs


# ===================================================
# sibling pair 生成
# ===================================================
def make_preference_pairs(comp_name, trajs, min_diff=0.0):
    groups = defaultdict(list)
    pairs = []

    for t in trajs:
        key = tuple(t.parent)
        groups[key].append(t)

    for key, group in groups.items():
        group = sorted(group, key=lambda t: t.world_value, reverse=True)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                d = group[i].world_value - group[j].world_value
                if d >= min_diff:
                    pairs.append({
                        "competition": comp_name,
                        "winner": "->".join(group[i].state),
                        "loser":  "->".join(group[j].state),
                        "winner_s": group[i].world_value,
                        "loser_s": group[j].world_value,
                        "score_diff": d,
                    })
    return pairs


# ===================================================
# MCTS
# ===================================================
def mcts(comp_name, root_state, reward_fn, get_possible_actions,
         max_depth=6, num_simulations=1000, c=1.41):

    root = MCTSNode(root_state)

    for _ in range(num_simulations):
        node = root
        depth = 0

        while node.children and depth < max_depth:
            node = node.best_child(c)
            depth += 1

        actions, p_scores = get_possible_actions(comp_name, node.state)

        for action, ps in zip(actions, p_scores):
            child = MCTSNode(
                state=node.state + [action],
                parent=node
            )
            node.children[action] = child

            value = reward_fn(comp_name, child, ps)

            n = child
            while n is not None:
                n.visits += 1
                n.value += value
                n = n.parent

    # get best path
    seq = root.state
    node = root
    for _ in range(max_depth):
        if not node.children:
            break
        node = node.best_child(c=0)
        seq = node.state

    return root, seq


# ===================================================
# LLM actions
# ===================================================
def parse_llm_json(text):
    import re
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return json.loads(text)


def get_actions(comp_name, state):
    history = "->".join(state)
    text = get_response(comp_name, history)

    try:
        data = parse_llm_json(text)
    except json.JSONDecodeError:
        print("⚠️ LLM JSON 解析失败：", text)
        return [], []

    return (
        [x["hypothesis"] for x in data if "hypothesis" in x],
        [x["potential_score"] for x in data if "potential_score" in x],
    )


# ===================================================
# Reward Model
# ===================================================
reward_model_path = "/data/userdata/v-lijingyuan/dpo/dpo1/last_run_7/last_run_7"
reward_base_model = "Qwen/Qwen3-4B"
competition_mapping_path = "/data/userdata/v-lijingyuan/train_reward_stage2/comp_to_scen.json"

# tokenizer = AutoTokenizer.from_pretrained(reward_base_model)
# if not getattr(tokenizer, "pad_token", None):
#     tokenizer.pad_token = tokenizer.eos_token

# model = RewardModelInference(
#     base_model_name=reward_base_model,
#     adapter_path=os.path.join(reward_model_path, "lora_adapter"),
#     reward_head_path=os.path.join(reward_model_path, "reward_head.pt"),
# ).to("cuda")
# model.eval()

with open(competition_mapping_path, "r") as f:
    comp_dict = json.load(f)


def reward_fn(comp_name, child, ps):
    s = "->".join(child.state)
    rewards = model.compute_reward([s], tokenizer, comp_dict[comp_name])

    score1 = rewards[0] / (1 + rewards[0])
    score2 = ps / 10
    final = rewards[0]  # 0.7 * score1 + 0.3 * score2

    child.ps = ps
    child.world_values = final
    return final


# ===================================================
# 单比赛任务（多进程入口）
# ===================================================
def run_one_competition(comp_name, gpu_id):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # ================= CUDA & MODEL init inside subprocess =================
    tokenizer = AutoTokenizer.from_pretrained(reward_base_model)
    if not getattr(tokenizer, "pad_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    model = RewardModelInference(
        base_model_name=reward_base_model,
        adapter_path=os.path.join(reward_model_path, "lora_adapter"),
        reward_head_path=os.path.join(reward_model_path, "reward_head.pt"),
    ).to("cuda")
    model.eval()

    with open(competition_mapping_path, "r") as f:
        comp_dict = json.load(f)

    def reward_fn(comp_name, child, ps):
        s = "->".join(child.state)
        rewards = model.compute_reward([s], tokenizer, comp_dict[comp_name])

        score1 = rewards[0] / (1 + rewards[0])
        score2 = ps / 10
        final = rewards[0]  # 0.7 * score1 + 0.3 * score2

        child.ps = ps
        child.world_values = final
        return final
    # ======================================================================

    root, best_seq = mcts(
        comp_name, [],
        reward_fn, get_actions,
        max_depth=6,
        num_simulations=40,
        c=1.41
    )

    trajs = collect_trajectories(root)
    pairs = make_preference_pairs(comp_name, trajs)

    append_jsonl(TRAJ_LOG, [traj_to_dict(t) for t in trajs])
    append_jsonl(PAIR_LOG, pairs)

    print(f"[{comp_name}] DONE | {len(trajs)} trajs, {len(pairs)} pairs")


# ===================================================
# 主入口：多比赛并行
# ===================================================
if __name__ == "__main__":
    from multiprocessing import set_start_method

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass  # 上下文已经设置了，直接忽略

    competitions = list(comp_dict.keys())
    gpu_slots = [0, 1, 2, 3]

    # 给每个比赛分配 GPU
    tasks = [(comp, gpu_slots[i % len(gpu_slots)]) for i, comp in enumerate(competitions)]

    def wrapper(args):
        comp, gpu = args
        run_one_competition(comp, gpu)

    with Pool(processes=4) as pool:  # 最多同时跑 4 个进程
        pool.map(wrapper, tasks)
