import torch
from torch.utils.data import Dataset, DataLoader, random_split
import datasets
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def create_tokenizer(raw_dataset, vocab_size, min_frequency, save_path=None):
    """"""

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
    )
    tokenizer.pre_tokenizer = Whitespace()
    iter_dataset = [i for i in raw_dataset["train"]["ch"]] + [
        i for i in raw_dataset["train"]["en"]
    ]
    tokenizer.train_from_iterator(iter_dataset, trainer)
    if save_path is not None:
        tokenizer.save(save_path)


def return_tokenized(raw_dataset, tokenizer: Tokenizer, max_len):
    """返回token化后的数据"""
    # 设置tokenizer的行为
    # token的模板
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    # padding
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), length=max_len)
    # truncation
    tokenizer.enable_truncation(max_length=max_len)
    raw_dataset_en = raw_dataset["train"]["en"]
    raw_dataset_ch = raw_dataset["train"]["ch"]
    # tokenized_en = tokenizer.encode_batch(raw_dataset_en, add_special_tokens=True)
    # tokenized_ch = tokenizer.encode_batch(raw_dataset_ch, add_special_tokens=True)
    # 批量token化
    batch_size = int(len(raw_dataset_en) / 100)
    tokenized_en = []
    tokenized_ch = []
    for i in range(0, len(raw_dataset_en), batch_size):
        tokenized_en += tokenizer.encode_batch(
            raw_dataset_en[i : min(i + batch_size, len(raw_dataset_en))],
            add_special_tokens=True,
        )
        tokenized_ch += tokenizer.encode_batch(
            raw_dataset_ch[i : min(i + batch_size, len(raw_dataset_en))],
            add_special_tokens=True,
        )
    return tokenized_en, tokenized_ch


class CustomDataset(Dataset):
    def __init__(self, tokenized_en, tokenized_ch):
        super().__init__()
        self.tokenized_en = tokenized_en
        self.tokenized_ch = tokenized_ch

    def __len__(self):
        return (
            len(self.tokenized_en)
            if len(self.tokenized_en) == len(self.tokenized_ch)
            else None
        )

    def __getitem__(self, i):

        return {
            "src_ids": torch.tensor(self.tokenized_en[i].ids),
            "src_mask": torch.tensor(self.tokenized_en[i].attention_mask),
            "tar_ids": torch.tensor(self.tokenized_ch[i].ids),
            "tar_mask": torch.tensor(self.tokenized_ch[i].attention_mask),
        }


def return_dataloader(
    tokenized_en, tokenized_ch, ratio, batch_size: int, shuffle: bool
):
    """返回训练，验证，测试的dataloader"""
    dataset_torch = CustomDataset(tokenized_en=tokenized_en, tokenized_ch=tokenized_ch)
    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset_torch, ratio, generator=torch.Generator().manual_seed(42)
    )
    train_dataloader, val_dataloader, test_dataloader = [
        DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        for dataset in (train_dataset, val_dataset, test_dataset)
    ]

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    import os

    # 从hugging face下载
    file_name = "zetavg/coct-en-zh-tw-translations-twp-300k"
    save_dir = "./output"
    save_tokenzier = save_dir + "/" + "tokenizer.json"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 加载数据集
    raw_dataset = load_dataset(file_name)
    print(raw_dataset.keys())
    print(raw_dataset)

    if not os.path.exists(save_tokenzier):
        create_tokenizer(
            raw_dataset, vocab_size=8000, min_frequency=4, save_path=save_tokenzier
        )
    print(type(raw_dataset))

    tokenizer = Tokenizer.from_file(save_tokenzier)

    tokenized_en, tokenized_ch = return_tokenized(
        raw_dataset=raw_dataset, tokenizer=tokenizer, max_len=200
    )
    # # 分析数据集
    # ids_len_en = []
    # ids_len_ch = []
    # for i in range(len(tokenized_en)):
    #     ids_len_en += [len(tokenized_en[i].ids)]
    #     ids_len_ch += [len(tokenized_ch[i].ids)]
    # print(
    #     "en_max:",
    #     max(ids_len_en),
    #     "en_min:",
    #     min(ids_len_en),
    #     "en_average:",
    #     sum(ids_len_en) / len(ids_len_en),
    # )
    # print(
    #     "ch_max:",
    #     max(ids_len_ch),
    #     "ch_min:",
    #     min(ids_len_ch),
    #     "ch_average:",
    #     sum(ids_len_ch) / len(ids_len_ch),
    # )
    # print(tokenized_en[0])
    # print(tokenized_en[0].ids)
    # print(tokenized_en[0].attention_mask)
    # 测试return_dataloader
    # dataset_torch = CustomDataset(tokenized_en=tokenized_en, tokenized_ch=tokenized_ch)
    # dataloader = DataLoader(dataset=dataset_torch, batch_size=2, shuffle=True)
    # for i in dataloader:
    #     print(type(i["src_ids"]))
    #     break
