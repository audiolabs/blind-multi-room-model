from typing import Union
import math
import pickle
import torch
from torch import nn
from torch import Tensor
from einops import rearrange

from src.models.components.cnn import CNNEncoder


class Transformer(nn.Module):
    """Transformer encoder for speech, uses CNN front-end"""

    def __init__(
        self,
        front_end: CNNEncoder,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        out_features: int,
        state: Union[str, None] = None,
    ) -> None:
        super().__init__()

        self.front_end = front_end

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.tf = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.tf_map = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

        # EDF decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=dim_feedforward),
            nn.GELU(),
            nn.Linear(in_features=dim_feedforward, out_features=out_features),
            nn.Tanh(),
        )

        # load trained states
        if state is not None:
            with open(state, "rb") as f:
                state = pickle.load(f)
            self.load_state_dict(state["component_state"])
            print("State successfully loaded.")

    def forward(self, x: Tensor) -> Tensor:
        # compute direct output
        tokens = self.front_end(x)
        # reshape for transformer
        tokens = rearrange(tokens, "b c f t -> b t (c f)")
        features = self.tf(self.pe(tokens))
        # temporal pooling and linear mapping
        latents = self.tf_map(torch.mean(features, dim=1))
        # decode edcs from latents
        return self.decoder(latents)


class PositionalEncoding(nn.Module):
    """Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
