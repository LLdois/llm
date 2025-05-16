"""训练文件"""

from datasets import load_dataset
import os
from data import create_tokenizer, return_tokenized, return_dataloader
from tokenizers import Tokenizer
import model
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import Optimizer
from torch.nn import Module
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from dataclasses import dataclass

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def trainer(
    model: Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fun: torch.nn.CrossEntropyLoss,
    device: torch.device,
    max_steps: int = 100000,
    eval_interval: int = 5000,
    save_path: str = "best_model.pth",
):
    model.to(device)
    # model = torch.compile(model)
    torch.set_float32_matmul_precision("high")
    step = 0
    running_loss = 0.0
    best_val_loss = float("inf")  # 初始化最佳验证损失为无穷大

    model.train()
    data_iterator = iter(train_dataloader)  # 创建数据迭代器

    with tqdm(total=max_steps, desc="Training", unit="step") as pbar:
        while step < max_steps:
            try:
                batch = next(data_iterator)
            except StopIteration:
                # 数据集遍历完后重置迭代器
                data_iterator = iter(train_dataloader)
                batch = next(data_iterator)

            src_ids, tgt_ids, src_padding_mask, tgt_padding_mask = (
                batch["src_ids"].to(device),
                batch["tgt_ids"].to(device),
                batch["src_padding_mask"].to(device),
                batch["tgt_padding_mask"].to(device),
            )

            optimizer.zero_grad()

            # import code

            # code.interact(local=locals())

            # 模型前向传播
            with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
                logits = model(
                    src_ids,
                    tgt_ids[:, 0:-1],
                    src_padding_mask,
                    tgt_padding_mask[:, 0:-1],
                )
            # 计算损失
            loss = loss_fun(
                logits.view(-1, logits.size(-1)), tgt_ids[:, 1:].contiguous().view(-1)
            )
            print(f"loss:{loss.item()}")
            loss.backward()

            clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            running_loss += loss.item()
            step += 1
            pbar.update(1)

            # 每隔 eval_interval 步进行一次验证
            if step % eval_interval == 0 or step == max_steps:
                avg_train_loss = (
                    running_loss / eval_interval
                    if step != max_steps
                    else running_loss / (step % eval_interval or eval_interval)
                )
                print(f"Step {step}/{max_steps} - Training loss: {avg_train_loss:.4f}")
                running_loss = 0.0  # 重置运行损失

                model.eval()
                val_loss = evaluate(model, val_dataloader, loss_fun, device)
                print(f"Step {step}/{max_steps} - Validation loss: {val_loss:.4f}")

                # 检查是否是最佳模型并保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), save_path)
                    print(
                        f"New best model saved at step {step} with validation loss: {best_val_loss:.4f}"
                    )

                model.train()  # 恢复训练模式


def evaluate(
    model: Module,
    val_dataloader: DataLoader,
    loss_fun: torch.nn.CrossEntropyLoss,
    device: torch.device,
):
    model.to(device)
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            src_ids, tgt_ids, src_padding_mask, tgt_padding_mask = (
                batch["src_ids"].to(device),
                batch["tgt_ids"].to(device),
                batch["src_padding_mask"].to(device),
                batch["tgt_padding_mask"].to(device),
            )
            # 模型前向传播
            logits = model(
                src_ids, tgt_ids[:, 0:-1], src_padding_mask, tgt_padding_mask[:, 0:-1]
            )
            # 计算损失
            loss = loss_fun(
                logits.view(-1, logits.size(-1)), tgt_ids[:, 1:].contiguous().view(-1)
            )
            running_loss += loss.item()

    avg_val_loss = running_loss / len(val_dataloader)
    return avg_val_loss


if __name__ == "__main__":
    # 超参数
    # model config
    @dataclass
    class config:
        vocab_size: int = 32000
        max_len: int = 256 + 1
        N: int = 6
        d_model: int = 512
        n_head: int = 8
        p_drop: float = 0.1

    # train config
    @dataclass
    class train_config:
        hugging_face_dataset: str = "zetavg/coct-en-zh-tw-translations-twp-300k"
        save_dir: str = "./output"
        vocab_size: int = 32000

        min_frequency: int = 6
        save_tokenzier: str = os.path.join(save_dir, "tokenizer.json")
        train_ratio = (0.9, 0.1)
        batch_size: int = 64

        shuffle = True

        max_steps = 400000
        eval_interval = 1000

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        check_point_path = os.path.join(save_dir, "model.pth")
        pre_check_point_path = os.path.join(save_dir, "pre.pth")

    if not os.path.exists(train_config.save_dir):
        os.mkdir(train_config.save_dir)

    # 加载原始数据集
    raw_dataset = load_dataset(train_config.hugging_face_dataset)
    print(raw_dataset.keys())
    print(raw_dataset["train"][0:3])
    # 基于原始数据训练tokenizer(BPE)
    if not os.path.exists(train_config.save_tokenzier):
        create_tokenizer(
            raw_dataset,
            vocab_size=train_config.vocab_size,
            min_frequency=train_config.min_frequency,
            save_path=train_config.save_tokenzier,
        )
    # 加载训练好的tokenizer
    tokenizer = Tokenizer.from_file(train_config.save_tokenzier)
    # token化数据
    tokenized_en, tokenized_ch = return_tokenized(
        raw_dataset=raw_dataset, tokenizer=tokenizer, max_len=config.max_len
    )
    # 划分数据集并返回dataloader
    train_dataloader, val_dataloader = return_dataloader(
        tokenized_en,
        tokenized_ch,
        train_config.train_ratio,
        train_config.batch_size,
        train_config.shuffle,
    )
    # 定义transformer模型
    transformer = model.Transformer(config)
    # 加载预训练权重
    if os.path.exists(train_config.pre_check_point_path):
        transformer.load_state_dict(torch.load(train_config.pre_check_point_path))
    # 优化器
    optimizer = optim.AdamW(transformer.parameters(), lr=1e-6, weight_decay=0.01)
    # 损失函数
    loss_fun = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    trainer(
        transformer,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fun,
        train_config.device,
        train_config.max_steps,
        train_config.eval_interval,
        train_config.check_point_path,
    )
