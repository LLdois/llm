"""在测试集上评估模型的最终性能"""

from tokenizers import Tokenizer
import model
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn.functional as F

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def inference(
    model: Module,
    src_string: str,
    device: torch.device,
    max_len: int = 200,
    save_tokenizer: str = "./output/tokenizer.json",
):
    model.eval()
    with torch.no_grad():
        # 将源序列转换为model能处理的ids
        # 加载tokenizer
        tokenizer = Tokenizer.from_file(save_tokenizer)
        # padding
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id("[PAD]"), length=max_len - 1
        )
        # truncation
        tokenizer.enable_truncation(max_length=max_len - 1)
        src_token = tokenizer.encode(src_string)
        src_ids = torch.tensor(src_token.ids).unsqueeze(0).to(device)
        src_padding_mask = (
            torch.tensor(src_token.attention_mask).unsqueeze(0).to(device)
        )
        # 制作目标序列
        tgt_string = "[BOS]"
        tgt_ids = torch.tensor([1]).view(1, -1).to(device)
        encoder_output_list = model.encoder(src_ids, src_padding_mask)

        for i in range(0, max_len - 2):
            decoder_output_list = model.decoder(
                encoder_output_list, tgt_ids, src_padding_mask=src_padding_mask
            )
            logits = model.output_linear(decoder_output_list[-1])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            tgt_ids = torch.cat((tgt_ids, xcol), dim=1)
            if xcol.item() == 2:
                break
        print(tgt_ids)
        tgt_string = tokenizer.decode(tgt_ids.cpu().tolist()[0])
        print(f"tgt_string->>>{tgt_string},token_len->>>{tgt_ids.size(-1)}")


if __name__ == "__main__":
    # 超参数
    from dataclasses import dataclass
    import os

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

        min_frequency: int = 4
        save_tokenizer: str = os.path.join(save_dir, "tokenizer.json")
        train_ratio = (0.9, 0.1)
        batch_size: int = 256

        shuffle = True

        max_steps = 50000
        eval_interval = 5000

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        check_point_path = os.path.join(save_dir, "model.pth")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 定义transformer模型
    transformer = model.Transformer(config)
    transformer.load_state_dict(torch.load(train_config.check_point_path))
    transformer.to(device)
    # src_string = "While the China Times Group has decided to drop its loss-making evening paper, it has acquired a majority stake in CtiTV, a Taiwan cable TV operator, and has also expressed an interest in acquiring China Television Company (CTV), which would make the group Taiwan's biggest multimedia empire."
    src_string = 'Civilian art, which is characterized by its non-academic foundation, developed in conjunction with the "liberalization of knowledge" trend kickstarted by Taiwan\'s community colleges in 1998. In fact, it was these community institutions that opened up a dialogue between Civilian Art and the academy, and Xizhi Community College that first incorporated Civilian Art into its curriculum'
    # src_string = "art"
    # print(f"src_string->>>{src_string}")
    inference(
        transformer, src_string, device, config.max_len, train_config.save_tokenizer
    )
