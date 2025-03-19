from datasets import load_dataset
import os
from data import create_tokenizer
from tokenizers import Tokenizer
from data import return_tokenized, return_dataloader
import model

if __name__ == "__main__":
    # 从hugging face下载
    # 超参数
    file_name = "zetavg/coct-en-zh-tw-translations-twp-300k"
    save_dir = "./output"
    save_tokenzier = save_dir + "/" + "tokenizer.json"

    vocab_size = 8000
    d_model = 512
    d_ff = 2048
    num_head = 8

    min_frequency = 4
    max_len = 200
    ratio = (0.7, 0.1, 0.2)
    batch_size = 32
    shuffle = True

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 加载原始数据集
    raw_dataset = load_dataset(file_name)
    print(raw_dataset.keys())
    print(raw_dataset["train"][0:3])
    # 基于原始数据训练tokenizer（BPE）
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
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_head,
    )
    # for input in train_dataloader:
    #     print(input["src_ids"].shape)
    #     print(input["src_mask"].shape)
    #     break
    input = next(iter(train_dataloader))
    output = transformer(input["src_ids"], input["src_padmask"])
