import math
import torch
import torch.nn as nn

import math
import torch
from torch import nn
import torch.nn.functional as F


class GPT(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, mlp_dim, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.vocab_size, self.n_embd),
            pos_emb=PositionalEncoding(self.n_embd, max_len=self.block_size),
            drop=nn.Dropout(self.dropout),
            h=nn.ModuleList([
                TransformerBlock(
                    d_model=self.n_embd,
                    n_heads=self.n_head,
                    d_ff=self.mlp_dim,
                    dropout=self.dropout
                ) for _ in range(self.n_layer)
            ]),
            ln_f=nn.LayerNorm(self.n_embd),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # Weight tying between embedding and lm_head
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.pos_emb.pe.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.block_size}")

        # Positional and token embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.pos_emb(tok_emb)  # (b, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Output logits
        logits = self.lm_head(x)

        # Optionally compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx