"""A tabular MLP combining learned category embeddings with numerical features."""

import torch
from torch import nn

from src.nn.model import CustomModel


class EmbeddingModel(nn.Module):
    """Embed each category column independently and concatenate it with scaled numbers."""

    def __init__(
        self,
        in_features: int,
        categorical_indices: list[int],
        numerical_indices: list[int],
        categorical_cardinalities: list[int],
        hidden_features: list[int],
        embedding_dims: int | list[int] = 4,
        out_features: int = 1,
        activation: str = "relu",
        batch_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        indices = categorical_indices + numerical_indices
        if sorted(indices) != list(range(in_features)):
            raise ValueError("Categorical and numerical indices must partition the input columns.")
        if not categorical_indices or len(categorical_indices) != len(categorical_cardinalities):
            raise ValueError("Each categorical column needs its own cardinality.")
        dims = (
            [embedding_dims] * len(categorical_indices)
            if isinstance(embedding_dims, int)
            else embedding_dims
        )
        if len(dims) != len(categorical_indices) or any(dim < 1 for dim in dims):
            raise ValueError("Embedding dimensions must be positive and match categorical columns.")
        if any(size < 1 for size in categorical_cardinalities):
            raise ValueError("Category cardinalities must include the unknown-category slot.")

        self.categorical_indices = categorical_indices
        self.numerical_indices = numerical_indices
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(size, dim, padding_idx=0)
                for size, dim in zip(categorical_cardinalities, dims, strict=True)
            ]
        )
        self.mlp = CustomModel(
            in_features=len(numerical_indices) + sum(dims),
            hidden_features=hidden_features,
            out_features=out_features,
            activation=activation,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert integral category IDs to LongTensor indices and return binary logits."""
        inputs = [x[:, self.numerical_indices]]
        inputs.extend(
            embedding(x[:, index].long())
            for index, embedding in zip(self.categorical_indices, self.embeddings, strict=True)
        )
        return self.mlp(torch.cat(inputs, dim=1))
