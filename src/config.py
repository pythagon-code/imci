from dataclasses import dataclass

@dataclass
class Config:
    # FNN parameters
    fnn_hidden_size: int = 100
    fnn_num_layers: int = 5

    # Transformer parameters
    transformer_hidden_size: int = 100
    transformer_num_layers: int = 5
    transformer_embed_dim: int = 100
    transformer_num_heads: int = 10

    # Extra option
    num_modules: int = 100