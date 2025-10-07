# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=2048, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Create transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Embed and add positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        # Transformer forward pass
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary size
        return self.output_projection(output)

def create_padding_mask(seq, pad_token=0):
    """Create padding mask for sequences"""
    return (seq == pad_token)

def create_look_ahead_mask(size):
    """Create look-ahead mask for decoder"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 1

# Example usage
if __name__ == "__main__":
    # Model parameters
    vocab_size = 1000
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    # Create model
    model = TransformerModel(vocab_size)
    
    # Create sample data
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))
    
    # Create masks
    src_padding_mask = create_padding_mask(src)
    tgt_padding_mask = create_padding_mask(tgt)
    tgt_mask = create_look_ahead_mask(tgt_len)
    
    # Forward pass
    output = model(src, tgt, 
                   src_key_padding_mask=src_padding_mask,
                   tgt_key_padding_mask=tgt_padding_mask,
                   tgt_mask=tgt_mask)
    
    print(f"Input source shape: {src.shape}")
    print(f"Input target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
