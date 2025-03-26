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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用device:{device}")
    model = GPT(GPTConfig)
    model.eval()
    model.to(device)

    import tiktoken

    num_return_sequences = 5
    max_length = 30

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

    x = tokens.to("cuda")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            topk_probs, topk_indices = torch.topk(F.softmax(logits, -1), 50)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=-1)
    for i in range(num_return_sequences):
        tokens = x[i, :].tolist()
        print(">", tokenizer.decode(tokens))
