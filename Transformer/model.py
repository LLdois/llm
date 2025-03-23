"""定义transformer模型"""

import torch
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, device: torch.device, max_len: int, d_model: int):
        """
        初始化位置编码。

        Args:
            device (torch.device): 设备(CPU或GPU)。
            max_len (int): 模型可输入序列的最大长度。
            d_model (int): Transformer 的隐藏层维度。
        """
        super().__init__()
        self.pos = torch.zeros((1, max_len, d_model), device=device)
        temp = torch.arange(max_len).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, d_model, 2) / d_model
        )
        self.pos[:, :, 0::2] = torch.sin(temp)
        self.pos[:, :, 1::2] = torch.cos(temp)

    def forward(
        self, input_embedding: torch.tensor, padding_mask: torch.tensor = None
    ) -> torch.Tensor:
        """
        为输入的 embedding 添加位置编码。

        Args:
            input_embedding (torch.Tensor):shape (batch_size, max_len, d_model) 的输入 embedding。
            padding_mask (torch.Tensor, optional):shape (batch_size, max_len) 的 padding mask。

        Returns:
            torch.Tensor: 添加了位置信息的 embedding。
        """
        in_emb_pos = input_embedding + self.pos
        # 对于padding部分不添加位置编码,屏蔽掉
        if padding_mask is not None:
            padding_mask = padding_mask == 0
            in_emb_pos = in_emb_pos.masked_fill(padding_mask.unsqueeze(2), 0)
            # print(in_emb_pos[0][-1])
            # print(in_emb_pos[0][0])
        return in_emb_pos


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(
        self,
        device: torch.device,
        d_model: int,
        num_head: int,
        is_causal_mask: bool = False,
    ):
        """
        初始化多头注意力机制。

        Args:
            device (torch.device): 设备(CPU或GPU)。
            d_model (int): Transformer 的隐藏层维度。
            num_head (int): 多头注意力的头数。
            is_causal_mask (bool, optional): 是否使用因果掩码 (解码器使用)。默认为 False。
        """
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.num_head = num_head
        self.is_causal_mask = is_causal_mask

        self.q_matrix = nn.Linear(d_model, d_model, bias=False)
        self.k_matrix = nn.Linear(d_model, d_model, bias=False)
        self.v_matrix = nn.Linear(d_model, d_model, bias=False)
        self.o_matrix = nn.Linear(d_model, d_model, bias=False)

    def _mask_softmax(
        self, attention_scores: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            q_raw (torch.Tensor): 查询张量 (query),shape为 (batch_size, max_len, d_model)。
            kv_raw (torch.Tensor): 键和值张量 (key & value),shape为 (batch_size, max_len, d_model)。
            padding_mask (torch.Tensor, optional):
                - 填充掩码,shape为 (batch_size, max_len)。
                - 其中 1 表示有效位置,0 表示填充位置(mask 掉）。

        Returns:
            torch.Tensor:
                - shape为 (batch_size, max_len, d_model) 的多头注意力输出张量。
        """

        if padding_mask is None:
            return F.softmax(attention_scores, -1)
        # 调整 maskshape,使其匹配 attention_scores
        padding_mask = padding_mask.repeat_interleave(self.num_head, dim=0)
        # padding_mask.shape(batch_size*num_head,max_len)
        padding_mask = padding_mask.unsqueeze(1) == 0
        # padding_mask.shape(batch_size*num_head,1,max_len)
        attention_scores = attention_scores.masked_fill(padding_mask, -1e9)
        # 添加因果掩码
        if self.is_causal_mask:
            mask = torch.ones(
                (attention_scores.size(-2), attention_scores.size(-1)),
                device=self.device,
            )
            mask = mask.triu(diagonal=1)
            mask = mask.unsqueeze(0) == 1
            attention_scores = attention_scores.masked_fill(mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        注意力机制

        Args:
            q (torch.Tensor): 查询张量 (query),shape为 (batch_size, max_len, d_model)。
            k (torch.Tensor): 键张量 (key),shape为 (batch_size, max_len, d_model)。
            v (torch.Tensor): 值张量 (value),shape为 (batch_size, max_len, d_model)。
            padding_mask (torch.Tensor, 可选):
                - 目标序列的填充掩码,shape为 (batch_size, max_len)。
                - 其中 1 表示有效位置,0 表示填充位置(mask 掉）。

        Returns:
            torch.Tensor:
                - shape为 (batch_size, max_len, d_model) 的注意力输出张量。
        """

        attention_scores = torch.bmm(q, torch.transpose(k, 1, 2)) / torch.sqrt(
            torch.tensor(self.d_model / self.num_head)
        )
        attention_weights = self._mask_softmax(attention_scores, padding_mask)
        return torch.bmm(attention_weights, v)

    def _trans_qkv(self, concat_qkv: torch.Tensor) -> torch.Tensor:
        """
        由于多个头的qkv为了并行变换concat在一起了,此函数将其分割

        Args:
            concat_qkv (torch.Tensor):
                - shape为 (batch_size, max_len, d_model),即多个头的 QKV 拼接在一起的形式。

        Returns:
            torch.Tensor:
                - shape为 (batch_size * num_head, max_len, d_model / num_head)
        """
        temp = concat_qkv.reshape(
            concat_qkv.shape[0], concat_qkv.shape[1], self.num_head, -1
        )
        # temp.shape(batch_size,max_len,num_head,d_model/num_head)
        temp = temp.permute(0, 2, 1, 3)
        # temp.shape(batch_size,num_head,max_len,d_model/num_head)
        return temp.reshape(-1, temp.shape[2], temp.shape[3])

    def _concat_head_atten(self, head_atten: torch.Tensor) -> torch.Tensor:
        """
        对head_atten的shape做变换,达到将多个头的注意力结果concat在一起的结果。

        Args:
            head_atten (torch.Tensor):
                - 多头注意力计算后的结果。
                - shape为 (batch_size * num_head, max_len, d_model / num_head)

        Returns:
            torch.Tensor:
                - shape为 (batch_size, max_len, d_model),即恢复到 Transformer 需要的shape。
        """
        temp = head_atten.reshape(
            -1, self.num_head, head_atten.shape[1], head_atten.shape[2]
        )
        # temp.shape(batch_size,num_head,max_len,d_model/num_head)
        temp = temp.permute(0, 2, 1, 3)
        # temp.shape(batch_size,max_len,,num_head,d_model/num_head)
        return temp.reshape(temp.shape[0], temp.shape[1], -1)

    def forward(
        self,
        q_raw: torch.Tensor,
        kv_raw: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        计算多头注意力。

        Args:
            q_raw (torch.Tensor): 查询张量shape (batch_size, max_len, d_model)。
            kv_raw (torch.Tensor): 键和值张量shape (batch_size, max_len, d_model)。
            padding_mask (torch.Tensor, optional):shape (batch_size, max_len) 的 padding mask。

        Returns:
            torch.Tensor: 计算后的注意力值shape (batch_size, max_len, d_model)。
        """
        # 进行线性变换,得到num_head个头的qkv信息
        q, k, v = (
            self.q_matrix(q_raw),
            self.k_matrix(kv_raw),
            self.v_matrix(kv_raw),
        )
        # 执行注意力机制操作
        head_atten = self.attention(
            self._trans_qkv(q), self._trans_qkv(k), self._trans_qkv(v), padding_mask
        )
        # head_atten.shape(batch_size*num_head,max_len,d_model/num_head)
        # 将多个头的注意力结果concat在一起之后再做一次线性变换
        return self.o_matrix(self._concat_head_atten(head_atten))


class PositionWiseFfn(nn.Module):
    """逐位置前馈神经网络"""

    def __init__(self, d_model: int, d_ff: int):
        """
        初始化前馈神经网络。

        Args:
            d_model (int): Transformer 宽度,即输入和输出的隐藏层特征数。
            d_ff (int): 前馈网络 (FFN) 的隐藏层维度
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        计算前馈神经网络的输出。

        Args:
            X (torch.Tensor): 输入张量shape (batch_size, max_len, d_model)。

        Returns:
            torch.Tensor:shape (batch_size, max_len, d_model) 的输出张量。
        """
        return self.linear2(self.relu(self.linear1(X)))


class EncoderBlock(nn.Module):
    """第i个Transformer编码器块,i=0,1,2,3...,N-1,用于对源序列进行特征提取和表示学习。"""

    def __init__(
        self,
        device: torch.device,
        d_model: int,
        d_ff: int,
        num_head: int,
        p_drop: float,
    ):
        """
        初始化编码器块。

        Args:
            device (torch.device): 设备(CPU或GPU)。
            d_model (int): Transformer 宽度,即输入和输出的隐藏层特征数。
            d_ff (int): 前馈网络 (FFN) 的隐藏层维度
            num_head (int): 多头注意力 (Multi-Head Attention) 机制中的注意力头数。
            p_drop (float): Dropout 比率,用于正则化训练,防止过拟合。
        """
        super().__init__()
        self.p_drop = p_drop
        self.device = device
        self.mh_attention_layer = MultiHeadAttention(self.device, d_model, num_head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.pw_ffn_linear = PositionWiseFfn(d_model=d_model, d_ff=d_ff)

    def forward(
        self, encoder_output: torch.Tensor, src_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            encoder_output (torch.Tensor):
                - 当 i == 0 时,为源序列 (source sequence) 的嵌入表示 (src_embedding)。
                - 当 i > 0 时,为上一个编码器块的输出。
                - shape为 (batch_size, max_len, d_model)。
            src_padding_mask (torch.Tensor, optional):
                - 源序列的填充掩码,标识填充部分 (padding positions),防止其影响注意力计算。
                - shape为 (batch_size, max_len)。
                - 若为 None,则不进行填充掩码计算。

        Returns:
            torch.Tensor:
                - 当前编码器块的输出,shape为 (batch_size, max_len, d_model)。
        """
        temp = self.layer_norm1(
            encoder_output
            + F.dropout(
                self.mh_attention_layer(
                    q_raw=encoder_output,
                    kv_raw=encoder_output,
                    padding_mask=src_padding_mask,
                ),
                p=self.p_drop,
            )
        )
        encoder_output = self.layer_norm2(
            temp + F.dropout(self.pw_ffn_linear(temp), p=self.p_drop)
        )
        return encoder_output


class DecoderBlock(nn.Module):
    """第i个Transformer解码器块,i=0,1,2,3...,N-1"""

    def __init__(
        self,
        device: torch.device,
        d_model: int,
        d_ff: int,
        num_head: int,
        p_drop: float,
    ):
        """
        初始化解码器块。

        Args:
            device (torch.device): 设备(CPU或GPU)。
            d_model (int): Transformer 宽度,即输入和输出的隐藏层特征数。
            d_ff (int): 前馈网络 (FFN) 的隐藏层维度
            num_head (int): 多头注意力 (Multi-Head Attention) 机制中的注意力头数。
            p_drop (float): Dropout 比率,用于正则化训练,防止过拟合。
        """
        super().__init__()
        self.device = device
        # 解码器块需要添加因果掩码
        self.causal_mask = True
        self.p_drop = p_drop
        # caucal注意力
        self.causal_mh_attention_layer = MultiHeadAttention(
            self.device, d_model, num_head, is_causal_mask=self.causal_mask
        )
        # 交叉注意力,与编码器的注意力计算方式一样,只是q来自于解码器,k和v来自编码器
        self.cross_mh_attention_layer = MultiHeadAttention(
            self.device, d_model, num_head
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.pw_ffn_linear = PositionWiseFfn(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        encoder_output: torch.Tensor,
        decoder_output: torch.Tensor,
        tgt_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output (torch.Tensor): 编码器 (Encoder) 的输出,shape为 (batch_size, max_len, d_model)。
            decoder_output (torch.Tensor): 上一个解码器块的输出：
                - 当 i > 0 时,为第 i-1 个解码器块的输出。
                - 当 i == 0 时,为目标序列 (tgtget sequence) 的嵌入表示 (tgt_embedding)。
                shape为 (batch_size, max_len, d_model)。
            tgt_padding_mask (torch.Tensor, optional): 目标序列的填充掩码,标识填充部分 (padding positions),
                shape为 (batch_size, max_len)。如果为 None,则不进行填充掩码计算。

        Returns:
            torch.Tensor: 当前解码器块的输出,shape为 (batch_size, max_len, d_model)。
        """
        temp = self.layer_norm1(
            decoder_output
            + F.dropout(
                self.causal_mh_attention_layer(
                    q_raw=decoder_output,
                    kv_raw=decoder_output,
                    padding_mask=tgt_padding_mask,
                ),
                p=self.p_drop,
            )
        )
        temp = self.layer_norm2(
            temp
            + F.dropout(
                self.cross_mh_attention_layer(
                    q_raw=encoder_output,
                    kv_raw=decoder_output,
                    padding_mask=tgt_padding_mask,
                ),
                p=self.p_drop,
            )
        )
        decoder_output = self.layer_norm3(
            temp + F.dropout(self.pw_ffn_linear(temp), p=self.p_drop)
        )
        return decoder_output


class Transformer(nn.Module):
    """Transformer model"""

    def __init__(
        self,
        device: torch.device,
        vocab_size: int,
        max_len: int,
        N: int,
        d_model: int,
        d_ff: int,
        num_head: int,
        p_drop: float,
        is_autorge: bool = False,
    ):
        """
        初始化 Transformer 模型。

        Args:
            device (torch.device): 设备(CPU或GPU)。
            vocab_size (int): 词汇表大小。
            max_len (int): 模型可接受的最大序列长度。
            N (int): 编码器和解码器块的数量。
            d_model (int): Transformer 的隐藏层维度。
            d_ff (int): 前馈网络的隐藏层维度。
            num_head (int): 多头注意力的头数。
            p_drop (float): Dropout 率。
        """
        super().__init__()
        self.device = device
        self.N = N
        self.p_drop = p_drop

        self.embedding_layer = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_layer = PositionalEncoding(
            device=self.device, max_len=max_len, d_model=d_model
        )

        self.encoder_seq = nn.Sequential()
        self.decoder_seq = nn.Sequential()

        for i in range(N):
            self.encoder_seq.add_module(
                "encoder_block" + str(i),
                EncoderBlock(
                    self.device,
                    d_model=d_model,
                    d_ff=d_ff,
                    num_head=num_head,
                    p_drop=self.p_drop,
                ),
            )
            self.decoder_seq.add_module(
                "decoder_block" + str(i),
                DecoderBlock(
                    device,
                    d_model=d_model,
                    d_ff=d_ff,
                    num_head=num_head,
                    p_drop=self.p_drop,
                ),
            )

        self.output_linear = nn.Linear(d_model, vocab_size)

    def encoder(self, src_ids: torch.Tensor, src_padding_mask: torch.Tensor) -> list:
        """
        Args:
            src_ids (torch.Tensor): 源序列的 token idsshape (batch_size, max_len)。
            src_padding_mask (torch.Tensor, optional): 源序列的 padding mask。
        Returns:
            encoder_output_list (list):encoder每层的隐藏状态
        """
        # 经过编码器
        src_embedding = self.embedding_layer(src_ids)
        src_emb_pos = self.pos_layer(src_embedding, src_padding_mask)
        # 添加dropout
        src_emb_pos = F.dropout(src_emb_pos, self.p_drop)
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
        tgt_padding_mask: torch.Tensor,
    ) -> list:
        """
        Args:
            encoder_output_list (list):encoder每层的隐藏状态
            tgt_ids (torch.Tensor): 目标序列的 token idsshape (batch_size, max_len)。
            tgt_padding_mask (torch.Tensor, optional): 目标序列的 padding mask。
        Returns:
            decoder_output_list (list):decoder每层的隐藏状态
        """
        # 经过解码器
        tgt_embedding = self.embedding_layer(tgt_ids)
        tgt_emb_pos = self.pos_layer(tgt_embedding, tgt_padding_mask)
        # 添加dropout
        tgt_emb_pos = F.dropout(tgt_emb_pos, p=self.p_drop)
        # 经过N个解码器块
        decoder_output_list = []
        for i, decoder_block in enumerate(self.decoder_seq):
            if i == 0:
                decoder_output_list.append(
                    decoder_block(encoder_output_list[i], tgt_emb_pos, tgt_padding_mask)
                )
            else:
                decoder_output_list.append(
                    decoder_block(
                        encoder_output_list[i],
                        decoder_output_list[i - 1],
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
        """
        前向传播。

        Args:
            src_ids (torch.Tensor): 源序列的 token idsshape (batch_size, max_len)。
            tgt_ids (torch.Tensor): 目标序列的 token idsshape (batch_size, max_len)。
            src_padding_mask (torch.Tensor, optional): 源序列的 padding mask。
            tgt_padding_mask (torch.Tensor, optional): 目标序列的 padding mask。

        Returns:
            torch.Tensor: 返回最终每个token属于每个vocab中token的logits
        """
        encoder_output_list = self.encoder(src_ids, src_padding_mask)
        decoder_output_list = self.decoder(
            encoder_output_list, tgt_ids, tgt_padding_mask
        )
        return self.output_linear(decoder_output_list[-1])


if __name__ == "__main__":
    input = torch.tensor([[5, 2, 1, 0, 0]])
    transformer = Transformer(
        vocab_size=8000,
        max_len=10,
        N=2,
        d_model=512,
        d_ff=2048,
        num_head=8,
    )
    print(input.shape)
    output = transformer(input)
    print(output)
    print(output.shape)
