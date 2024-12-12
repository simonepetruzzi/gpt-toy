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


class CausalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        #Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Output linear layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Dropout layer
        self.attn_dropout = nn.Dropout(dropout)

        def forward(self, x):
            batch_size, seq_length, embed_dim = x.size()

            # Apply projections to query, keys and values
            Q = self.q_proj(x)
            K = self.k_proj(x) 
            V = self.v_proj(x)  

            # Reshape and transpose for multi-head attention
            # New shape: (batch_size, num_heads, seq_length, head_dim)
            Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

            # Compute scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Create a causal mask to ensure that each position can only attend to previous positions
            # Mask shape: (1, 1, seq_length, seq_length)
            mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))

            # Apply softmax to get attention probabilities
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)  # Apply dropout

            # Weighted sum of values
            # Output shape: (batch_size, num_heads, seq_length, head_dim)
            attn_output = torch.matmul(attn_probs, V)

            # Concatenate heads and project to output dimension
            # First, transpose to (batch_size, seq_length, num_heads, head_dim)
            attn_output = attn_output.transpose(1, 2).contiguous()
            # Then, reshape to (batch_size, seq_length, embed_dim)
            attn_output = attn_output.view(batch_size, seq_length, embed_dim)

            # Final linear projection
            output = self.out_proj(attn_output)  # (batch_size, seq_length, embed_dim)

            return output



class TransformerBlock():
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Transformer block as it is in the Attention is all you need paper
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(hidden_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_output)
        # Feed-forward
        ff_output = self.ff(self.ln2(x))
        x = x + self.dropout(ff_output)
        return x
