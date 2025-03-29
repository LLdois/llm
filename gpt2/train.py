from dataclasses import dataclass
import torch
import torch.nn as nn
import tiktoken
from torch.nn import functional as F
from model import GPT


# 超参数
@dataclass
class GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073


# 学习率调度器
def get_lr(it):
    import math

    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@dataclass
class TrainConfig:
    batch_size: int = 4
    seq_len: int = 32


import numpy as np


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_process, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_process = num_process
        assert split in {"train", "val"}

        # 加载切片文件
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"切片文件为：{shards}"
        if master_process:
            print(f"{split}中有{len(shards)}个切片文件")

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_process
        if self.current_position + (B * T * self.num_process + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.current_shard)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y


if __name__ == "__main__":
    # ddp
    # torchrun --standalone --nproc_per_node=2 train.py
    import os
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "cuda不可用"
        # 初始化进程组
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        # apple的后端
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"使用设备：{device}")

    # 梯度累计
    # 一个batch的token数
    total_batch_size = 524288
    B = 8
    T = 1024
    assert (
        total_batch_size % (ddp_world_size * B * T) == 0
    ), "确保总token数能被ddp_world_size * B * T整除"
    grad_accum_step = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"原文中一个batchsize的总token数：{total_batch_size}")
        print(f"需要做梯度累计的step:{grad_accum_step}")

    # 确保可以复现
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    # 创建模型
    model = GPT(GPTConfig(vocab_size=50304))
    model.eval()
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    import time

    # 使用tf32精度训练
    torch.set_float32_matmul_precision("high")

    train_loader = DataLoaderLite(
        B, T, process_rank=ddp_rank, num_process=ddp_world_size, split="train"
    )
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8
    # )
    # 定制AdamW
    optimizer = raw_model.configure_optimizers(0.1, 3e-8, device)
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0
        for mini_step in range(grad_accum_step):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = mini_step == grad_accum_step - 1
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_step
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for pa in optimizer.param_groups:
            pa["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (
            train_loader.B * train_loader.T * grad_accum_step * ddp_world_size
        ) / (t1 - t0)
        if master_process:
            print(
                f"step:{step},loss:{loss_accum.item()},norm:{norm.item():.2f},lr:{lr:.8f},dt:{dt:.2f}ms,tokens/sec:{tokens_per_sec:.2f}"
            )
    if ddp:
        destroy_process_group()
