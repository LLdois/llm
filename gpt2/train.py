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


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


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
        self.reset()

    def reset(self):
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
    B = 16
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
    from hellaswag import iterate_examples, render_example

    # 使用tf32精度训练
    torch.set_float32_matmul_precision("high")

    train_loader = DataLoaderLite(
        B, T, process_rank=ddp_rank, num_process=ddp_world_size, split="train"
    )
    val_loader = DataLoaderLite(
        B, T, process_rank=ddp_rank, num_process=ddp_world_size, split="val"
    )
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8
    # )
    # 定制AdamW
    optimizer = raw_model.configure_optimizers(0.1, 3e-8, device)
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime

    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y_%m_%d_%H:%M")
    log_file = os.path.join(log_dir, f"{current_time_str}_log.txt")
    # with open(log_file, "w") as f:
    #     pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1
        # evaluate in hellaswag
        if step % 10 == 0 or last_step:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(
                    num_correct_norm, dtype=torch.long, device=device
                )
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(
                    f"Hellaswag accuracy:{num_correct_norm}/{num_total}={acc_norm:.4f}"
                )
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # val loss
        if step % 10 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    val_loss_accum += loss.detach()
                val_loss_accum = val_loss_accum / val_loss_steps
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"val loss:{val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # gengrate
        if (step > 0 and step % 20 == 0) or last_step:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokenizer = tiktoken.get_encoding("gpt2")
            tokens = tokenizer.encode("Hello, I'am a language model,")
            tokens = (
                torch.tensor(tokens, dtype=torch.long)
                .unsqueeze(0)
                .repeat(num_return_sequences, 1)
            )
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    logits, loss = model(xgen)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = tokenizer.decode(tokens)
                print(f"rank:{ddp_rank},sample:{i}:{decoded}")

        # train
        model.train()
        optimizer.zero_grad()
        loss_accum = 0
        # 梯度累计
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
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(
                    log_dir, f"model_{step:05d}_{current_time_str}"
                )
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                torch.save(checkpoint, checkpoint_path)
    if ddp:
        destroy_process_group()
