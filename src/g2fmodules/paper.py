import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class modelPaper(nn.Module):
    """Simple Transformer encoder-decoder wrapper.

    This implements embeddings + positional encoding + PyTorch Transformer
    and a final linear layer to project to `phone_size` logits.

    Notes:
    - During training call forward(src, tgt) where `src` and `tgt` are LongTensors
      of token ids with shape (batch, seq_len).
    - For inference you can either provide a `tgt` (teacher forcing) or
      implement a generation loop outside this module.
    """

    def __init__(
        self,
        vocab_size: int,
        phone_size: int,
        embed_dim: int,
        hidden_dim: int,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = hidden_dim

        self.src_embedding = nn.Embedding(vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(phone_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)

        # Use PyTorch Transformer (encoder-decoder)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.output_fc = nn.Linear(embed_dim, phone_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.LongTensor, tgt: torch.LongTensor) -> torch.Tensor:
        """Forward pass.

        Args:
            src: (batch, src_len) source token ids
            tgt: (batch, tgt_len) target token ids (teacher forcing)

        Returns:
            logits: (batch, tgt_len, phone_size)
        """
        # Embedding + positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Create masks: transformer expects tgt_mask for causal decoding
        tgt_seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

        # No src_key_padding_mask or tgt_key_padding_mask provided here —
        # if you have padding tokens, pass masks to the transformer call.
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)

        logits = self.output_fc(output)
        return logits