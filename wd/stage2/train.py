import os, json, argparse, math
from typing import List, Dict, Any, Optional
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from lightning_fabric.utilities.seed import seed_everything
 
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
 
from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger


# =========================
# Utils
# =========================
 
def robust_stats(arr: List[float]):
    """返回均值mu与标准差sigma（sigma用MAD近似、避免极端值影响）。"""
    x = np.array(arr, dtype=np.float64)
    mu = float(np.mean(x))
    mad = float(np.median(np.abs(x - np.median(x))) + 1e-8)
    sigma = 1.4826 * mad  # 正态下MAD->std的系数
    if sigma < 1e-6:
        sigma = float(np.std(x) + 1e-6)
    return mu, sigma
 
 
 
# =========================
# Dataset
# =========================
class RewardDataset(Dataset):
    def __init__(
        self,
        pairs: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        use_margin: bool = False,
        mu: float = 0.0,
        sigma: float = 1.0,
        alpha: float = 2.0,
        beta: float = 0.5,
        system_prompt: Optional[str] = None,
        comp_to_scen: Optional[Dict[str, str]] = None,
    ):
        self.pairs = pairs
        self.tok = tokenizer
        if not getattr(self.tok, "pad_token", None):
            self.tok.pad_token = self.tok.eos_token
        self.max_length = max_length
        self.use_margin = use_margin
        self.mu, self.sigma, self.alpha, self.beta = mu, sigma, alpha, beta
        self.comp_to_scen = comp_to_scen 
        self.system_prompt = system_prompt or (
            "You are a senior data science competition judge and solution expert.\n"
            "Your task is to evaluate the quality, reasoning progression, and innovation of hypothesis chains.\n"
            "A hypothesis chain shows iterative improvement of solutions.\n"
            "You should assess:\n"
            "1) reasoning correctness and consistency across steps,\n"
            "2) improvement and refinement through the chain,\n"
            "3) final hypothesis quality and practicality.\n"
            "Be strict and fair. Provide expert-level insight."
        )

 
    def __len__(self):
        return len(self.pairs)
 
    def _enc(self, s: str,comp_name: str):
        #s = f"{self.system_prompt} Solution: {s}{self.tok.eos_token}"
        scen_text = self.comp_to_scen[comp_name]
        s = (
            f"{self.system_prompt}\n\n"
            f"Competition description:\n{scen_text}\n\n"
            "Hypothesis Chain (each step separated by '->'):\n"
            f"{s}\n\n"
            "<think>\n"
            "Analyze the evolution of hypotheses, step-by-step, identifying strengths, weaknesses, and logical progression.\n"
            "Focus on clarity, correctness, and improvement.\n"
            "Make sure to consider the chain direction from earliest to latest.\n"
            "</think>\n\n"
            "Final Evaluation:\n"
        )
        enc = self.tok(
            s,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
 
    @staticmethod
    def weight_from_diff(diff: float) -> float:
        """
        单调递增、上界2.0的权重映射（稳健、免标定）：
        w = 0.5 + 1.5 * (|d| / (|d| + 1))
        """
        d = abs(diff if diff is not None else 0.0)
        return 0.5 + 1.5 * (d / (d + 1.0))
 
    def margin_from_diff(self, diff: float) -> float:
        """
        目标边际：对score_diff做z-score，再tanh压缩到[-alpha,alpha]
        """
        if diff is None:
            return 0.0
        z = (diff - self.mu) / max(self.sigma, 1e-6)
        return float(self.alpha * math.tanh(self.beta * z))
 
    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        s = p["comptation_name"]
        wi, wm = self._enc(p["winner"],s)
        li, lm = self._enc(p["loser"],s)
        sd = float(p.get("score_diff", 0.0))
        #true_winner = 1 if p.get("score_a", 0.0) >= p.get("score_b", 0.0) else 0

        item = {
            "winner_input_ids": wi,
            "winner_attention_mask": wm,
            "loser_input_ids": li,
            "loser_attention_mask": lm,
        }
        item = {
            "winner_input_ids": wi,
            "winner_attention_mask": wm,
            "loser_input_ids": li,
            "loser_attention_mask": lm,
            "weight": torch.tensor(self.weight_from_diff(sd), dtype=torch.float32),
            "score_diff_raw": torch.tensor(sd, dtype=torch.float32),  # 验证期做相关分析/标定用
            #"true_winner": torch.tensor(true_winner, dtype=torch.long)
        }
        if self.use_margin:
            item["target_margin"] = torch.tensor(
                self.margin_from_diff(sd), dtype=torch.float32
            )
        return item
 
 
# =========================
# Model (GPT-2 + LoRA + reward_head)
# =========================
class RewardModelPL(pl.LightningModule):
    def __init__(
        self,
        model_name="gpt2",
        lr=1e-5,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        tau=1.0,
        loss_type="bt",  # bt | bt_reg | hinge
        lambda_reg=0.0,  # bt_reg 有效
        m0=0.0,          # hinge 基础边际
        gamma=1.0,       # hinge 权重斜率
        use_bf16=False,
    ):
        super().__init__()
        self.save_hyperparameters()
 
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = False
 
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"]#, "k_proj", "o_proj", "mlp.fc1", "mlp.fc2"]

        )
        self.model = get_peft_model(base, peft_config)
 
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
 
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        # hidden size
        hs = getattr(self.model.config, "hidden_size",
                    getattr(self.model.config, "n_embd",
                    getattr(self.model.config, "d_model", None)))
        if hs is None:
            # fallback
            hs = self.model.embed_tokens.embedding_dim
        self.reward_head = nn.Linear(hs, 1)
 
        self.lr = lr
        self.tau = tau
        self.loss_type = loss_type
        self.lambda_reg = lambda_reg
        self.m0 = m0
        self.gamma = gamma
        self.train_acc =Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        # 缓存验证期指标
        self._val_pred_margins: List[float] = []
        self._val_score_diffs: List[float] = []

    def load_pretrained(self, ckpt_dir: str):
        print(f"[INFO] Loading pretrained reward model from {ckpt_dir}")

        # 1️⃣ Load LoRA adapter
        lora_path = os.path.join(ckpt_dir, "lora_adapter")
        if os.path.exists(lora_path):
            self.model.load_adapter(lora_path, adapter_name="default")
            print("[INFO] Loaded LoRA adapter.")
        else:
            raise FileNotFoundError(f"LoRA adapter not found at {lora_path}")

        # 2️⃣ Load reward head
        rh_path = os.path.join(ckpt_dir, "reward_head.pt")
        if os.path.exists(rh_path):
            state = torch.load(rh_path, map_location="cpu")
            self.reward_head.load_state_dict(state)
            print("[INFO] Loaded reward head.")
        else:
            raise FileNotFoundError(f"reward_head.pt not found at {rh_path}")

    @staticmethod
    def pool_last_nonpad(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden: [B,T,H], attn_mask: [B,T]
        lengths = attn_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
        return last_hidden.gather(1, idx).squeeze(1)
 
    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = out.hidden_states[-1]
        pooled = self.pool_last_nonpad(last_hidden, attention_mask)
        reward = self.reward_head(pooled).squeeze(-1)  # [B]
        reward = reward
        return reward
 
    def _margin(self, rw, rl):
        return (rw - rl) / self.tau
 
    def _loss_batch(self, batch):
        rw = self(batch["winner_input_ids"], batch["winner_attention_mask"])
        rl = self(batch["loser_input_ids"], batch["loser_attention_mask"])
        margin = self._margin(rw, rl)
        acc = (rw > rl).float().mean()
 
        if self.loss_type == "bt":
            loss = -(F.logsigmoid(margin)).mean()
 
        elif self.loss_type == "bt_reg":
            loss_cls = -(F.logsigmoid(margin) * batch["weight"]).mean()
            target_m = batch.get("target_margin", None)
            if target_m is None:
                raise ValueError("bt_reg 需要在Dataset中启用 use_margin=True 才会返回 target_margin")
            loss_reg = F.huber_loss(margin, target_m, delta=1.0, reduction="none")
            loss_reg = (loss_reg * batch["weight"]).mean()
            loss = loss_cls + self.lambda_reg * loss_reg
 
        elif self.loss_type == "hinge":
            margin_req = self.m0 + self.gamma * batch["weight"]        # m(w)
            loss = F.relu(margin_req - (rw - rl)).mean()
 
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
 
        return loss, acc, (rw, rl)
 
    def training_step(self, batch, idx):
        loss, _, (rw, rl) = self._loss_batch(batch)

        # prediction
        pred = (rw > rl).long()
        #true = batch["true_winner"]
        true = torch.ones_like(pred)   # 正确的 winner 永远是 winner

        self.train_acc.update(pred, true)

        self.log("train_loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss
        
    def on_train_epoch_end(self):
        acc_epoch = self.train_acc.compute()
        self.log("train_acc_epoch", acc_epoch, prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, idx):
        loss, _, (rw, rl) = self._loss_batch(batch)

        pred = (rw > rl).long()
        true = torch.ones_like(pred)   # 正确的 winner 永远是 winner

        self.val_acc.update(pred, true)

        self.log("val_loss_step", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        acc_epoch = self.val_acc.compute()
        self.log("val_acc_epoch", acc_epoch, prog_bar=True)
        self.val_acc.reset()
 
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
 
    def on_train_end(self):
        # 保存 LoRA 适配器和 reward_head + 校准系数
        logdir = None
        if isinstance(self.logger, list):
            for lg in self.logger:
                if hasattr(lg, "log_dir") and lg.log_dir is not None:
                    logdir = lg.log_dir
                    break
        elif hasattr(self.logger, "log_dir") and self.logger.log_dir is not None:
            logdir = self.logger.log_dir

        # fallback
        if logdir is None:
            logdir = "./logs/last_run_6"

        # 确保是字符串类型
        logdir = str(logdir)
        os.makedirs(logdir, exist_ok=True)

        # LoRA adapter
        self.model.save_pretrained(os.path.join(logdir, "lora_adapter"))
        # reward head
        torch.save(self.reward_head.state_dict(), os.path.join(logdir, "reward_head.pt"))
 
        # 保存一个简单的 param 报告
        trainable, total = 0, 0
        for p in self.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        with open(os.path.join(logdir, "param_report.txt"), "w") as f:
            f.write(f"trainable={trainable}, total={total}\n")
 
        # 如果有最后一次验证的 a,b,τ，持久化（从logger的指标里取最近值）
        a = float(self.trainer.callback_metrics.get("calib_a", torch.tensor(1.0)).item())
        b = float(self.trainer.callback_metrics.get("calib_b", torch.tensor(0.0)).item())
        calib = {"a": a, "b": b, "tau": float(self.tau)}
        with open(os.path.join(logdir, "calib.json"), "w") as f:
            json.dump(calib, f, indent=2)
        print(f"[INFO] Saved LoRA adapter, reward_head and calib to {logdir}")
       
 
# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=10000)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--use_bf16", action="store_true")
 
    # Loss&mapping 超参
    parser.add_argument("--loss_type", type=str, choices=["bt", "bt_reg", "hinge"], default="bt")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=2.0)  # margin范围
    parser.add_argument("--beta", type=float, default=0.5)   # margin压缩斜率
    parser.add_argument("--m0", type=float, default=0.0)     # hinge 基础边际
    parser.add_argument("--gamma", type=float, default=1.0)  # hinge 权重斜率
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default='last_run_5',
        help="Path to pretrained reward model checkpoint (folder with lora_adapter/ and reward_head.pt)"
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    import os
    os.environ["WANDB_API_KEY"] = "58a239a973f50ed918896b3de374988f4d26d86a"

    # 创建 Wandb logger
    wandb_logger = WandbLogger(
        project="reward_diff",  # 你在wandb上创建的project名
        name="rm_run2",                   # 可自定义每次训练的名字
        log_model=True                     # 自动记录模型权重
    )


    # 读数据
    # with open(args.json_path, "r", encoding="utf-8") as f:
    #     pairs: List[Dict[str, Any]] = json.load(f)

    # target_comps = {
    #     "tweet-sentiment-extraction",
    #     "whale-categorization-playground",
    #     "jigsaw-toxic-comment-classification-challenge"
    # }
    # filtered_pairs = [
    #     p for p in pairs
    #     if p.get("comptation_name") in target_comps
    # ]

    # print(f"✅ Loaded {len(filtered_pairs)} pairs after filtering "
    #     f"from {len(pairs)} total samples.")
    # print("Included competitions:", {p["comptation_name"] for p in filtered_pairs})
        
    # pairs = filtered_pairs

    with open("comp_to_scen.json", "r") as f:
        comp_to_scen = json.load(f)

    debug = False
    if debug:
        import ijson

        pairs = []
        with open(args.json_path, "r", encoding="utf-8") as f:
            for obj in ijson.items(f, 'item'):
                pairs.append(obj)
                if len(pairs) >= 1000:
                    break
    
    else:
        with open(args.json_path, "r", encoding="utf-8") as f:
            pairs: List[Dict[str, Any]] = json.load(f)

    # print(pairs)

    if not debug:
        target_comps = {
            "cassava-leaf-disease-classification",
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
            "hubmap-kidney-segmentation",
        }
        print("Total pairs loaded:", len(target_comps))
        filtered_pairs = [
            p for p in pairs
            if p.get("comptation_name") in target_comps
        ]

        print(f"✅ Loaded {len(filtered_pairs)} pairs after filtering "
            f"from {len(pairs)} total samples.")
        print("Included competitions:", {p["comptation_name"] for p in filtered_pairs})
            
        pairs = filtered_pairs


    # 统计 score_diff 的 mu/sigma（给 margin 映射用）
    diffs = [float(p.get("score_diff", 0.0)) for p in pairs]
    mu, sigma = robust_stats(diffs)
 
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if not getattr(tok, "pad_token", None):
        tok.pad_token = tok.eos_token
 
    use_margin = (args.loss_type == "bt_reg")
 
    full_dataset = RewardDataset(
        pairs=pairs,
        tokenizer=tok,
        max_length=args.max_length,
        use_margin=use_margin,
        mu=mu,
        sigma=sigma,
        alpha=args.alpha,
        beta=args.beta,
        comp_to_scen= comp_to_scen
    )
 
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    from torch import Generator

    g = Generator()
    g.manual_seed(args.seed)  
    train_set, val_set = random_split(full_dataset, [n_train, n_val],generator=g)
 
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
 
    model = RewardModelPL(
        model_name=args.model_name,
        lr=args.lr,
        tau=args.tau,
        loss_type=args.loss_type,
        lambda_reg=args.lambda_reg,
        m0=args.m0,
        gamma=args.gamma,
        use_bf16=args.use_bf16,
    )
    if args.pretrained_ckpt is not None:
        model.load_pretrained(args.pretrained_ckpt)
        print("load model!")
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name="tb")
    csv_logger = CSVLogger(save_dir=args.log_dir, name="csv")
 
    ckpt_cb = ModelCheckpoint(
        dirpath=args.log_dir,
        filename="best-rm",
        monitor="val_loss_step",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )
 
    strategy = DDPStrategy(find_unused_parameters=False)
 
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        max_epochs=args.max_epochs,
        precision=args.precision,
        gradient_clip_val=0.1,
        #logger=[tb_logger, csv_logger],
        callbacks=[ckpt_cb],
        log_every_n_steps=1,
        logger=[wandb_logger, tb_logger, csv_logger]
    )
    trainer.fit(model, train_loader, val_loader)
 
 
if __name__ == "__main__":
    main()