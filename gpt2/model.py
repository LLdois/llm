import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.c_proj.res_std = True
        self.register_buffer(
            "caucal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        qkv = self.c_attn(x)  # shape(batch_size,seq_len,n_embd*3)
        q, k, v = qkv.split(n_embd, dim=-1)  # shape(batch_size,seq_len,n_embd)
        q = q.view(batch_size, seq_len, self.config.n_head, -1).transpose(
            1, 2
        )  # shape(batch_size,n_head,seq_len,n_embd/n_head)
        k = k.view(batch_size, seq_len, self.config.n_head, -1).transpose(
            1, 2
        )  # shape(batch_size,n_head,seq_len,n_embd/n_head)
        v = v.view(batch_size, seq_len, self.config.n_head, -1).transpose(
            1, 2
        )  # shape(batch_size,n_head,seq_len,n_embd/n_head)
        # 普通attention
        # attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        # attention_scores = attention_scores.masked_fill(
        #     self.caucal_mask[0, 0, :seq_len, :seq_len] == 0, float("-inf")
        # )
        # attention_weights = F.softmax(
        #     attention_scores, -1
        # )  # shape(batch_size,n_head,seq_len,seq_len)
        # y = attention_weights @ v  ## shape(batch_size,n_head,seq_len,n_embd/n_head)

        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        y = self.c_proj(y)
        return y


class Mlp(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.c_proj.res_std = True

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = Mlp(config)

    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x


class GPT(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            res_std = 0.02
            if hasattr(module, "res_std"):
                res_std = (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=res_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        batch_size, seq_len = idx.shape
        assert (
            seq_len <= self.config.block_size
        ), f"小批量序列长度seq_len：{seq_len}不能大于{self.config.block_size}"
        pos = torch.arange(seq_len, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(
        self, weight_decay, learning_rate, device, is_master_process=False
    ):
        parm_dict = {pn: p for pn, p in self.named_parameters()}
        parm_dict = {pn: p for pn, p in parm_dict.items() if p.requires_grad}
        decay_params = [p for n, p in parm_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in parm_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weght_decay": weight_decay},
            {"params": nodecay_params, "weght_decay": 0},
        ]
        num_dacay_params = sum(p.numel() for p in decay_params)
        num_nodacay_params = sum(p.numel() for p in nodecay_params)
        if is_master_process:
            print(
                f"权重衰减的tensor数量:{len(decay_params)},共 {num_dacay_params:,} 个parameters"
            )
            print(
                f"没有权重衰减的tensor数量:{len(nodecay_params)},共 {num_nodacay_params:,} 个parameters"
            )
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # fused简单来说就是加速
        use_fused = fused_available and "cuda" in device
        if is_master_process:
            print(f"使用 AdamW fused:{use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        from dataclasses import dataclass

        @dataclass
        class GPTConfig:
            block_size: int = 1024
            n_layer: int = 12
            n_head: int = 12
            n_embd: int = 768
            vocab_size: int = 50257

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.caucal_mask")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
