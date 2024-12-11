import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
     def __init__(self, d_model, max_len=512):
        super().__init__()
        #Generate the encoding matrix 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #Populate the matrix with sinusoidal and cosinusoidal values creating encoding for each position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #Add batch dimension to pe, resulting in a shape of (1, max_len, d_model).
        pe = pe.unsqueeze(0)
        #Register "pe" as buffer such that it is not updated during training
        self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:, :x.size(1), :]
            return x


class MLP(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.0):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        out, weights = self.self_attention(x, x, x, need_weights=True, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return out



         

class TransformerBlock():
    pass