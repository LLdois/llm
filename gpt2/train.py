from dataclasses import dataclass
import torch
import torch.nn as nn
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


@dataclass
class TrainConfig:
    batch_size: int = 4
    seq_len: int = 32


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


if __name__ == "__main__":
    # 确保可以复现
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"使用device:{device}")
    model = GPT(GPTConfig)
    model.eval()
    model.to(device)

    import tiktoken

    num_return_sequences = 5
    max_length = 30

    # 加载hugginf face参数测试
    # model = GPT.from_pretrained("gpt2")
    # model.to(device)
    # tokenizer = tiktoken.get_encoding("gpt2")
    # tokens = tokenizer.encode("Hello, I'm a language model,")
    # tokens = torch.tensor(tokens, dtype=torch.long)
    # tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

    # x = tokens.to("cuda")

    # while x.size(1) < max_length:
    #     with torch.no_grad():
    #         logits = model(x)
    #         logits = logits[:, -1, :]
    #         topk_probs, topk_indices = torch.topk(F.softmax(logits, -1), 50)
    #         ix = torch.multinomial(topk_probs, 1)
    #         xcol = torch.gather(topk_indices, -1, ix)
    #         x = torch.cat((x, xcol), dim=-1)
    # for i in range(num_return_sequences):
    #     tokens = x[i, :].tolist()
    #     print(">", tokenizer.decode(tokens))

    # 莎士比亚数据集/测试
    # 1
    # batch_size, seq_len = 4, 32
    # with open("input.txt", "r") as f:
    #     text = f.read()
    # data = text[:1000]

    # tokenizer = tiktoken.get_encoding("gpt2")
    # ids = tokenizer.encode(data)
    # buf = torch.tensor(ids[: batch_size * seq_len + 1]).to(device)
    # x = buf[:-1].view(batch_size, seq_len)
    # y = buf[1:].view(batch_size, seq_len)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # for i in range(50):
    #     optimizer.zero_grad()
    #     logits, loss = model(x, y)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"step:{i},loss:{loss.item()}")

    # 2
    import time

    # 使用tf32精度训练
    torch.set_float32_matmul_precision("high")

    train_loader = DataLoaderLite(8, 1024)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(
            f"step:{i},loss:{loss.item()},dt:{dt:.2f}ms,tokens/sec:{tokens_per_sec:.2f}"
        )
