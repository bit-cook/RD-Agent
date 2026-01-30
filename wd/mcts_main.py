import math
import random
import matplotlib.pyplot as plt
import networkx as nx
from mcts import get_response
import json
from dataclasses import dataclass
import json
from collections import defaultdict
import json
import os
import uuid




def save_pairs(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

@dataclass
class Trajectory:
    state: list[str]
    parent: list[str]     # 新增
    world_value: float          
    visits: int


def traj_to_dict(traj):
    return {
        "state": traj.state,                    # list[str]
        "parent": traj.parent,
        "state_str": " -> ".join(traj.state),   # 方便人读 & pair
        "world_value": traj.world_value,   # 
        "visits": traj.visits,
        "length": len(traj.state),
    }
def save_trajectories(trajs, path):
    with open(path, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(traj_to_dict(t), ensure_ascii=False) + "\n")

class MCTSNode:
    def __init__(self, state=None, exp_feedback_list_desc=None, sota_exp_desc=None, parent=None):
        self.state = state or []  #
        self.exp_feedback_list_desc = exp_feedback_list_desc or []  #
        self.sota_exp_desc = sota_exp_desc  #
        self.parent = parent
        self.children = {}  
        self.visits = 0
        self.value = 0.0
        self.world_values = 0
        self.ps =0

    def is_fully_expanded(self, possible_actions):
        pa = {tuple(a) if isinstance(a, list) else a for a in possible_actions}
        ck = {tuple(k) if isinstance(k, list) else k for k in self.children.keys()}
        return pa <= ck


    def best_child(self, c=1.41):
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            if child.visits == 0:
                uct = float('inf')
            else:
                uct = child.value / child.visits + c * math.sqrt(math.log(self.visits) / child.visits)
            if uct > best_score:
                best_score = uct
                best_child = child
        return best_child
    


def collect_trajectories(root: MCTSNode):
    trajs = []

    def dfs(node: MCTSNode):
        # leaf -> 保存整条路径轨道
        if not node.children:
            trajs.append(
                Trajectory(
                    state=node.state.copy(),
                    parent=node.parent.state.copy() if node.parent else [],
                    world_value=float(node.world_values),
                    visits=node.visits,
                )
            )
        for child in node.children.values():
            dfs(child)

    dfs(root)
    return trajs

def make_preference_pairs(comp_name, trajs, min_diff=0.0):
    groups = defaultdict(list)
    pairs = []

    # ---- 只在 sibling 间比较 ----
    for t in trajs:
        key = tuple(t.parent)
        groups[key].append(t)

    for key, group in groups.items():
        group = sorted(group, key=lambda t: t.world_value, reverse=True)

        for i in range(len(group)):
            for j in range(i+1, len(group)):
                d = group[i].world_value - group[j].world_value
                if d >= min_diff:
                    pairs.append({
                        "competition": comp_name,
                        "winner": "->".join(group[i].state),
                        "loser":  "->".join(group[j].state),
                        "winner_s":group[i].world_value,
                        "loser_s": group[j].world_value,
                        "score_diff": d,
                    })

    return pairs

def mcts(comp_name,root_state, reward_fn, get_possible_actions, max_depth=6, num_simulations=1000, c=1.41):
    print(comp_name)
    root = MCTSNode(root_state)
    for _ in range(num_simulations):
        node = root
        depth = 0
        # 1. Selection
        while node.children and depth < max_depth:
            actions = get_possible_actions(comp_name,node.state)
            # if not node.is_fully_expanded(actions):
            #     break
            node = node.best_child(c)
            depth += 1

        actions, p_scores = get_possible_actions(comp_name, node.state)

        for action,ps in zip(actions,p_scores):
            child = MCTSNode(
                state=node.state + [action],
                parent=node
            )
            node.children[action] = child
            value = reward_fn(comp_name, child,ps)

            n = child
            while n is not None:
                n.visits += 1
                n.value += value
                n = n.parent

    sequence = root.state
    node = root
    for _ in range(max_depth):
        if not node.children:
            break
        node = node.best_child(c=0)  # c=0 表示只取最大平均价值
        sequence = node.state

    return root, sequence


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    """
    为树结构生成层次布局
    """
    if pos is None:
        pos = {root:(xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.successors(root))
    if len(children)!=0:
        dx = width/len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root)
    return pos


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    import json
    import re

    def parse_llm_json(text: str):
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())
        return json.loads(text)

    def get_actions(comp_name, state):
        history = " -> ".join(state)
        text = get_response(comp_name, history)

        try:
            data = parse_llm_json(text)
        except json.JSONDecodeError:
            print("⚠️ Hypothesis gen not JSON, fallback")
            print(text)
            return []

        return [
            item["hypothesis"]
            for item in data
            if isinstance(item, dict) and "hypothesis" in item
        ],[item["potential_score"]
            for item in data
            if isinstance(item, dict) and "potential_score" in item]


    from reward_inference import RewardModelInference
    from transformers import AutoTokenizer
    import os
    reward_model_path = "/data/userdata/v-lijingyuan/dpo/dpo1/last_run_7/last_run_7"
    reward_base_model = "Qwen/Qwen3-4B"
    competition_mapping_path = "/data/userdata/v-lijingyuan/train_reward_stage2/comp_to_scen.json"

    logdir = reward_model_path
    base_model = reward_base_model
    comp_dict_path = competition_mapping_path

    adapter_path = os.path.join(logdir, "lora_adapter")
    reward_head_path = os.path.join(logdir, "reward_head.pt")
    calib_path = os.path.join(logdir, "calib.json")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if not getattr(tokenizer, "pad_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    model = RewardModelInference(
            base_model_name=base_model,
            adapter_path=adapter_path,
            reward_head_path=reward_head_path,
        ).to("cuda")
    model.eval()
    with open(comp_dict_path, "r") as f:
        comp_dict = json.load(f)

    def reward_fn(comp_name,child,ps):
        state =child.state
        print(state)
        state = "->".join(state)
        print('aaaaaa')
        print(state)
        comp_description = comp_dict[comp_name]

        rewards = model.compute_reward(
            [state],
            tokenizer,
            comp_description
        )
        
        child.ps = ps 
        score1 = rewards[0]/(1+rewards[0])
        score2 = ps /10 
        final_score =0.7*score1 + 0.3*score2
        child.world_values = final_score#rewards[0]
        return  final_score#1#response.get("score", 0.0)

    comp_name = "iwildcam-2020-fgvc7"

    def get_final_data(comp_name):
        start = []
        root, best_seq = mcts(comp_name, start, reward_fn, get_actions, max_depth=6, num_simulations=15, c=1.41)
        trajs = collect_trajectories(root)
        pairs = make_preference_pairs(comp_name,trajs, min_diff=0.0)

        import os
        base_dir = "mcts_data"
        comp_dir = os.path.join(base_dir, comp_name)
        
        os.makedirs(comp_dir, exist_ok=True)
        save_pairs(
            pairs,
            os.path.join(comp_dir, "mcts_preference_pairs.jsonl")
        )

        save_trajectories(
            trajs,
            os.path.join(comp_dir, "mcts_trajectories.jsonl")
        )

    get_final_data(comp_name)



    # print("Best sequence found:", best_seq)

    # G = nx.DiGraph()
    # labels = {}
    # def state_to_id(state):
    #     return "ROOT" if not state else " -> ".join(state)
    # def add_edges(node):
    #     node_id = state_to_id(node.state)
    #     labels[node_id] = f"{node_id}\nV={node.value:.2f}\nN={node.visits}"

    #     for action, child in node.children.items():
    #         child_id = state_to_id(child.state)
    #         G.add_edge(node_id, child_id, label=action)
    #         add_edges(child)

    # add_edges(root)
    # pos = hierarchy_pos(G, root=state_to_id(start))

    # plt.figure(figsize=(12,6))
    # nx.draw(G, pos, with_labels=True, labels=labels, node_size=2500, node_color="lightblue", font_size=10)
    # edge_labels = nx.get_edge_attributes(G, 'label')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # plt.title("MCTS Tree Visualization (Hierarchical Layout)")
    # #plt.show()
    # plt.savefig("m1.png")
#dotenv -f /home/v-lijingyuan/rd-agent/RD-Agent/.env run -- python '/home/v-lijingyuan/MCTS_data/mcts_main.py'