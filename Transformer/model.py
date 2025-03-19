import torch
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos = torch.zeros((1, max_len, d_model))
        temp = torch.arange(max_len).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, d_model, 2) / d_model
        )
        self.pos[:, :, 0::2] = torch.sin(temp)
        self.pos[:, :, 1::2] = torch.cos(temp)

    def forward(self, input_embedding):
        in_emb_pos = input_embedding + self.pos[:, 0 : input_embedding.shape[1], :]
        return in_emb_pos


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head

        self.q_matrix = nn.Linear(d_model, d_model, bias=False)
        self.k_matrix = nn.Linear(d_model, d_model, bias=False)
        self.v_matrix = nn.Linear(d_model, d_model, bias=False)
        self.o_matrix = nn.Linear(d_model, d_model, bias=False)

    def mask_softmax(self, attention_weights, padding_mask=None):
        """遮蔽softmax"""
        if padding_mask is None:
            return F.softmax(attention_weights, -1)
        print(padding_mask.shape)

    def attention(
        self,
        q_head: torch.Tensor,
        k_head: torch.Tensor,
        v_head: torch.Tensor,
        padding_mask=None,
    ):
        """单头（普通）attention操作"""
        attention_scores = torch.bmm(
            q_head, torch.transpose(k_head, 1, 2)
        ) / torch.sqrt(torch.tensor(self.d_model / self.num_head))
        attention_weights = self.mask_softmax(attention_scores, padding_mask)
        return torch.bmm(attention_weights, v_head)

    def trans_qkv(self, concat_qkv: torch.Tensor):
        """由于多个头的qkv为了并行变换concat在一起了，此函数分割开便于调用普通attention"""
        # concat_qkv.shape(batch,seq_len,d_model)
        # return:shape(batch*num_head,seq_len,d_model/num_head)
        temp = concat_qkv.reshape(
            concat_qkv.shape[0], concat_qkv.shape[1], self.num_head, -1
        )
        # temp.shape(batch,seq_len,num_head,d_model/num_head)
        temp = temp.permute(0, 2, 1, 3)
        # temp.shape(batch,num_head,seq_len,d_model/num_head)
        return temp.reshape(-1, temp.shape[2], temp.shape[3])

    def trans_head_atten(self, head_atten: torch.Tensor):
        """此函数将head_atten转换为多个头的注意力层输出concat以后的形式"""
        # head_atten.shape(batch*num_head,seq_len,d_model/num_head)
        # rerurn:shape(batch,seq_len,d_model)
        temp = head_atten.reshape(
            -1, self.num_head, head_atten.shape[1], head_atten.shape[2]
        )
        # temp.shape(batch,num_head,seq_len,d_model/num_head)
        temp = temp.permute(0, 2, 1, 3)
        # temp.shape(batch,seq_len,,num_head,d_model/num_head)
        return temp.reshape(temp.shape[0], temp.shape[1], -1)

    def forward(self, embedding: torch.Tensor, padding_mask=None):
        # 这里的qkv是8个头的qkv合并在一起的，shape（batch,seq_len,num_head*(d_model/num_head))
        q, k, v = (
            self.q_matrix(embedding),
            self.k_matrix(embedding),
            self.v_matrix(embedding),
        )
        head_atten = self.attention(
            self.trans_qkv(q), self.trans_qkv(k), self.trans_qkv(v), padding_mask
        )
        # head_atten.shape(batch*num_head,seq_len,d_model/num_head)
        return self.o_matrix(self.trans_head_atten(head_atten))


class PositionWiseFfn(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, X: torch.Tensor):
        return self.linear2(self.relu(self.linear1(X)))


class EncodingBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, d_model, d_ff, num_head):
        super().__init__()
        self.mh_attention_layer = MultiHeadAttention(d_model, num_head)
        self.layer_norm = nn.LayerNorm(d_model)
        self.pw_ffn_linear = PositionWiseFfn(d_model=d_model, d_ff=d_ff)

    def forward(self, embedding: torch.Tensor, padding_mask=None):
        temp = self.layer_norm(
            embedding + self.mh_attention_layer(embedding, padding_mask)
        )
        return self.layer_norm(temp + self.pw_ffn_linear(temp))


class Transformer(nn.Module):
    """Transformer model"""

    def __init__(self, vocab_size, max_len, d_model, d_ff, num_head):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.enbedding_layer = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=0
        )
        self.pos_layer = PositionalEncoding(max_len=max_len, d_model=d_model)
        self.encoding_block = EncodingBlock(
            d_model=d_model, d_ff=d_ff, num_head=num_head
        )

    def forward(self, input, padding_mask=None):
        input_enbedding = self.enbedding_layer(input)
        in_emb_pos = self.pos_layer(input_enbedding)
        encoding_information = self.encoding_block(in_emb_pos, padding_mask)
        return input_enbedding


if __name__ == "__main__":
    input = torch.tensor([[5, 2, 1, 0, 0]])
    transformer = Transformer(
        vocab_size=8000,
        max_len=10,
        d_model=512,
        d_ff=2048,
        num_head=8,
    )
    print(input.shape)
    output = transformer(input)
    print(output)
    print(output.shape)
