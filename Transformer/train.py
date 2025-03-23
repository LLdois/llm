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
from torch.nn import Module, CrossEntropyLoss
import torch.nn as nn
from tqdm import tqdm


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
    """
    训练函数（基于批次更新,并保存最佳模型）

    Args:
        model (torch.nn.Module): 定义的Transformer模型。
        train_dataloader (DataLoader): 训练集的数据加载器。
        val_dataloader (DataLoader): 验证集的数据加载器。
        optimizer (torch.optim.Optimizer): 用于优化的优化器。
        loss_fun (CrossEntropyLoss): 损失函数,通常使用交叉熵损失。
        device (torch.device): 设备(CPU或GPU)。
        max_steps (int): 训练的最大步数,默认值为100000。
        eval_interval (int): 每隔多少步在验证集上评估一次,默认值为5000。
        save_path (str): 保存最佳模型的路径,默认值为"best_model.pth"。

    Returns:
        None
    """
    model.to(device)
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

            # 模型前向传播
            logits = model(
                src_ids, tgt_ids[:, 0:-1], src_padding_mask, tgt_padding_mask[:, 0:-1]
            )
            # 计算损失
            loss = loss_fun(
                logits.view(-1, logits.size(-1)), tgt_ids[:, 1:].contiguous().view(-1)
            )
            loss.backward()

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
    """
    在验证集上评估模型的表现。

    Args:
        model (torch.nn.Module): 定义的Transformer模型。
        val_dataloader (DataLoader): 验证集的数据加载器。
        loss_fun (CrossEntropyLoss): 损失函数,通常使用交叉熵损失。
        device (torch.device): 设备(CPU或GPU)。

    Returns:
        float: 平均验证损失。
    """
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
    # 从hugging face下载
    # 超参数
    file_name = "zetavg/coct-en-zh-tw-translations-twp-300k"
    save_dir = "./output"
    save_tokenzier = save_dir + "/" + "tokenizer.json"
    weights_path = "./output/best_model.pth"
    vocab_size = 8000
    N = 2
    d_model = 512
    d_ff = 2048
    num_head = 8
    p_drop = 0.1

    min_frequency = 4
    # 最大长度指的是tgt_ids的最大长度,src_ids = max_len - 1
    max_len = 200
    ratio = (0.7, 0.1, 0.2)
    batch_size = 32
    shuffle = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_steps = 100000
    eval_interval = 10000

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 加载原始数据集
    raw_dataset = load_dataset(file_name)
    print(raw_dataset.keys())
    print(raw_dataset["train"][0:3])
    # 基于原始数据训练tokenizer(BPE)
    if not os.path.exists(save_tokenzier):
        create_tokenizer(
            raw_dataset,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            save_path=save_tokenzier,
        )
    # 加载训练好的tokenizer
    tokenizer = Tokenizer.from_file(save_tokenzier)
    # token化数据
    tokenized_en, tokenized_ch = return_tokenized(
        raw_dataset=raw_dataset, tokenizer=tokenizer, max_len=max_len
    )
    # 划分数据集并返回dataloader
    train_dataloader, val_dataloader, test_dataloader = return_dataloader(
        tokenized_en, tokenized_ch, ratio, batch_size, shuffle
    )
    # 定义transformer模型
    transformer = model.Transformer(
        device=device,
        vocab_size=vocab_size,
        max_len=max_len - 1,
        N=N,
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_head,
        p_drop=p_drop,
    )
    # # 继续训练
    # if weights_path:
    #     transformer.load_state_dict(torch.load(weights_path))
    # 优化器
    optimizer = optim.AdamW(transformer.parameters(), lr=0.001, weight_decay=0.01)
    # 损失函数
    loss_fun = nn.CrossEntropyLoss()
    trainer(
        transformer,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fun,
        device,
        max_steps=100000,
        eval_interval=10000,
        save_path=save_dir + "/" + "best_model.pth",
    )
