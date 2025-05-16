"""定义transformer模型"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, config: dataclass):
        super().__init__()
        self.config = config
        self.pos = torch.zeros((1, config.max_len, config.d_model))
        temp = torch.arange(config.max_len).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, config.d_model, 2) / config.d_model
        )
        self.pos[:, :, 0::2] = torch.sin(temp)
        self.pos[:, :, 1::2] = torch.cos(temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        assert T <= self.config.max_len, "必须小于最大长度"
        self.pos = self.pos.to(x.device)
        self.pos = self.pos[:, :T, :]
        x = x + self.pos
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, config: dataclass, is_causal_mask=False):
        super().__init__()
        self.config = config
        self.is_causal_mask = is_causal_mask

        self.q_matrix = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_matrix = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_matrix = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_matrix = nn.Linear(config.d_model, config.d_model, bias=False)
        # causal_mask.shape(1,1,max_len,max_len)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.max_len, config.max_len)
            .tril(0)
            .unsqueeze(0)
            .unsqueeze(0),
        )

    def forward(
        self,
        q_raw: torch.Tensor,
        kv_raw: torch.Tensor,
        kv_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T_q, _ = q_raw.shape
        B, T_kv, _ = kv_raw.shape
        assert not (
            (T_kv != T_q) and self.is_causal_mask
        ), "交叉注意力层不可添加因果掩码"
        # 进行线性变换,得到n_head个头的qkv信息
        q, k, v = (
            self.q_matrix(q_raw),
            self.k_matrix(kv_raw),
            self.v_matrix(kv_raw),
        )
        # q,k,v.shape(B,T_q,d_model) or (B,T_kv,d_model)
        # 拆分头
        q = q.view(B, T_q, self.config.n_head, -1).transpose(1, 2)
        k = k.view(B, T_kv, self.config.n_head, -1).transpose(1, 2)
        v = v.view(B, T_kv, self.config.n_head, -1).transpose(1, 2)
        # shape(B,config.n_head,T_q,d_model/n_head) or (B,config.n_head,T_kv,d_model/n_head)

        # 注意力机制
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(
            self.config.d_model / self.config.n_head
        )
        # shape(B,config.n_head,T_q,T_kv)
        # 添加因果掩码
        if self.is_causal_mask:
            attention_scores = attention_scores.masked_fill(
                self.causal_mask[0, 0, :T_q, :T_q] == 0, float("-inf")
            )
        # 添加padding掩码
        if kv_padding_mask is not None:
            # shape(B,T_kv)
            kv_padding_mask = (
                kv_padding_mask.unsqueeze(1).repeat(1, T_q, 1).unsqueeze(1)
            )
            # shape(B,1,T_q,T_kv)
            attention_scores = attention_scores.masked_fill(
                kv_padding_mask == 0, float("-inf")
            )
        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, -1)

        y = attention_weights @ v
        # shape(B,n_head,T_q,d_model/n_head)
        y = y.transpose(2, 3).contiguous().view(B, T_q, -1)
        # shape(B,T_q,d_model)
        # 将多个头的注意力结果concat在一起之后再做一次线性变换
        y = self.o_matrix(y)
        return y


class PositionWiseFfn(nn.Module):
    """逐位置前馈神经网络"""

    def __init__(self, config: dataclass):

        super().__init__()
        self.config = config
        self.linear1 = nn.Linear(config.d_model, config.d_model * 4, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.d_model * 4, config.d_model, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        return self.linear2(self.relu(self.linear1(X)))


class EncoderBlock(nn.Module):
    """第i个Transformer编码器块,i=0,1,2,3...,N-1,用于对源序列进行特征提取和表示学习。"""

    def __init__(self, config: dataclass):

        super().__init__()
        self.config = config
        self.mh_attention_layer = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.pw_ffn_linear = PositionWiseFfn(config)

    def forward(
        self, encoder_output: torch.Tensor, src_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:

        temp = self.layer_norm1(
            encoder_output
            + F.dropout(
                self.mh_attention_layer(
                    q_raw=encoder_output,
                    kv_raw=encoder_output,
                    kv_padding_mask=src_padding_mask,
                ),
                p=self.config.p_drop,
            )
        )
        encoder_output = self.layer_norm2(
            temp + F.dropout(self.pw_ffn_linear(temp), self.config.p_drop)
        )
        return encoder_output


class DecoderBlock(nn.Module):
    """第i个Transformer解码器块,i=0,1,2,3...,N-1"""

    def __init__(self, config: dataclass):

        super().__init__()
        self.config = config
        # 解码器块需要添加因果掩码
        self.causal_mask = True
        # caucal注意力
        self.causal_mh_attention_layer = MultiHeadAttention(config, self.causal_mask)
        # 交叉注意力,与编码器的注意力计算方式一样,只是q来自于解码器,k和v来自编码器
        self.cross_mh_attention_layer = MultiHeadAttention(config)

        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.layer_norm3 = nn.LayerNorm(config.d_model)
        self.pw_ffn_linear = PositionWiseFfn(config)

    def forward(
        self,
        encoder_output: torch.Tensor,
        decoder_output: torch.Tensor,
        src_padding_mask: torch.Tensor = None,
        tgt_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        temp = self.layer_norm1(
            decoder_output
            + F.dropout(
                self.causal_mh_attention_layer(
                    q_raw=decoder_output,
                    kv_raw=decoder_output,
                    kv_padding_mask=tgt_padding_mask,
                ),
                p=self.config.p_drop,
            )
        )
        temp = self.layer_norm2(
            temp
            + F.dropout(
                self.cross_mh_attention_layer(
                    q_raw=temp,
                    kv_raw=encoder_output,
                    kv_padding_mask=src_padding_mask,
                ),
                p=self.config.p_drop,
            )
        )
        decoder_output = self.layer_norm3(
            temp + F.dropout(self.pw_ffn_linear(temp), self.config.p_drop)
        )
        return decoder_output


class Transformer(nn.Module):
    """Transformer model"""

    def __init__(self, config: dataclass):

        super().__init__()
        self.config = config
        self.embedding_layer = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=0
        )
        self.pos_layer = PositionalEncoding(config)

        self.encoder_seq = nn.Sequential()
        self.decoder_seq = nn.Sequential()

        for i in range(config.N):
            self.encoder_seq.add_module(
                "encoder_block" + str(i),
                EncoderBlock(config),
            )
            self.decoder_seq.add_module(
                "decoder_block" + str(i),
                DecoderBlock(config),
            )

        self.output_linear = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # output_linear和embedding共享权重
        self.output_linear.weight = self.embedding_layer.weight

    def encoder(self, src_ids: torch.Tensor, src_padding_mask: torch.Tensor) -> list:
        # 经过编码器
        src_embedding = self.embedding_layer(src_ids) * math.sqrt(self.config.d_model)
        src_emb_pos = self.pos_layer(src_embedding)
        # 添加dropout
        src_emb_pos = F.dropout(src_emb_pos, self.config.p_drop)
        # 经过N个编码器块
        encoder_output_list = []
        for i, encoder_block in enumerate(self.encoder_seq):
            if i == 0:
                encoder_output_list.append(encoder_block(src_emb_pos, src_padding_mask))
            else:
                encoder_output_list.append(
                    encoder_block(encoder_output_list[i - 1], src_padding_mask)
                )
        return encoder_output_list

    def decoder(
        self,
        encoder_output_list: list,
        tgt_ids: torch.Tensor,
        src_padding_mask: torch.Tensor = None,
        tgt_padding_mask: torch.Tensor = None,
    ) -> list:

        # 经过解码器
        tgt_embedding = self.embedding_layer(tgt_ids) * math.sqrt(self.config.d_model)
        tgt_emb_pos = self.pos_layer(tgt_embedding)
        # 添加dropout
        tgt_emb_pos = F.dropout(tgt_emb_pos, p=self.config.p_drop)
        # 经过N个解码器块
        decoder_output_list = []
        for i, decoder_block in enumerate(self.decoder_seq):
            if i == 0:
                decoder_output_list.append(
                    decoder_block(
                        encoder_output_list[-1],
                        tgt_emb_pos,
                        src_padding_mask,
                        tgt_padding_mask,
                    )
                )
            else:
                decoder_output_list.append(
                    decoder_block(
                        encoder_output_list[-1],
                        decoder_output_list[i - 1],
                        src_padding_mask,
                        tgt_padding_mask,
                    )
                )
        return decoder_output_list

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_padding_mask: torch.Tensor = None,
        tgt_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        encoder_output_list = self.encoder(src_ids, src_padding_mask)
        decoder_output_list = self.decoder(
            encoder_output_list, tgt_ids, src_padding_mask, tgt_padding_mask
        )
        return self.output_linear(decoder_output_list[-1])


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class config:
        vocab_size: int = 8000
        max_len: int = 10
        N: int = 2
        d_model: int = 512
        n_head: int = 8
        p_drop: float = 0.1

    input = torch.tensor([[4, 2, 5, 8, 0, 0], [4, 5, 85, 4, 2, 5]], dtype=torch.long)
    kv_padding_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long
    )
    transformer = Transformer(config)
    input = input.to("cuda")
    kv_padding_mask = kv_padding_mask.to("cuda")
    transformer.to("cuda")
    x = transformer(input, input, kv_padding_mask, kv_padding_mask)
    # embedding = nn.Embedding(8000, 512)
    # x = embedding(input)
    # pos = PositionalEncoding(config)
    # x = pos(x)
    # encoder = EncoderBlock(config)
    # en = encoder(x)
    # decoder = DecoderBlock(config)
    # x = decoder(en, x, kv_padding_mask)
    print(x)
    print(x.shape)
