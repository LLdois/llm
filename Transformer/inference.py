"""在测试集上评估模型的最终性能"""

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
import torch.nn.functional as F


def test(
    model: Module,
    src_string: str,
    device: torch.device,
    max_len: int = 200,
    save_tokenzier: str = "./output/tokenizer.json",
    test_dataloader: DataLoader = None,
):
    """
    在测试集上评估模型的表现（自回归推理）

    Args:
        model (torch.nn.Module): 定义的Transformer模型。
        test_dataloader (DataLoader): 验证集的数据加载器。
        device (torch.device): 设备(CPU或GPU)。
        weights_path (str): 模型权重路径。
        max_len (int): 如果模型性能不行,可能会无限生成,设置最大生成长度

    Returns:

    """
    model.eval()
    # 将源序列转换为model能处理的ids
    # 加载tokenizer
    tokenizer = Tokenizer.from_file(save_tokenzier)
    # padding
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), length=max_len - 1)
    # truncation
    tokenizer.enable_truncation(max_length=max_len - 1)
    src_token = tokenizer.encode(src_string)
    src_ids = torch.tensor(tokenizer.encode(src_string).ids).unsqueeze(0).to(device)
    src_padding_mask = torch.tensor(src_token.attention_mask).unsqueeze(0).to(device)
    # 制作目标序列
    tgt_string = "[BOS]"
    tgt_token = tokenizer.encode(tgt_string)
    tgt_ids = torch.tensor(tokenizer.encode(tgt_string).ids).unsqueeze(0).to(device)
    tgt_padding_mask = torch.tensor(tgt_token.attention_mask).unsqueeze(0).to(device)
    encoder_output_list = model.encoder(src_ids, src_padding_mask)
    for i in range(0, max_len - 2):
        decoder_output_list = model.decoder(
            encoder_output_list, tgt_ids, tgt_padding_mask
        )
        logits = model.output_linear(decoder_output_list[-1])
        tgt_ids[0, i + 1] = logits.argmax(dim=2)[0, i]
        tgt_padding_mask[0, i + 1] = 1
        if logits.argmax(dim=2)[0, i].item() == 2:
            break
    tgt_string = tokenizer.decode(tgt_ids.cpu().tolist()[0])
    print(tgt_string)


if __name__ == "__main__":
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_steps = 100000
    eval_interval = 10000

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
    transformer.load_state_dict(torch.load(weights_path))
    transformer.to(device)
    src_string = "who are you"
    test(transformer, src_string, device, max_len, save_tokenzier)
