import torch.nn as nn

from src.utils.nn import ModelOne as ModelOne


class CustomModel(nn.Module):
    """Configurable multilayer classifier with optional normalization and dropout."""

    def __init__(
            self,
            in_features: int,
            hidden_features: list[int],
            out_features: int = 1,
            activation: str = "relu",
            batch_norm: bool = False,
            dropout_rate: float = 0.0,
    ):
        """Assemble hidden layers from the configured widths and activation."""
        super().__init__()
        activation_map = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        if activation not in activation_map:
            raise AttributeError(f"Uknown activation function {activation}")
        
        layers: list[nn.Module] = []
        prev_feature = in_features

        for hidden_feature in hidden_features:
            layers.append(
                nn.Linear(in_features=prev_feature, out_features=hidden_feature))

            if batch_norm:
                layers.append(
                    nn.BatchNorm1d(num_features=hidden_feature)
                )

            layers.append(activation_map[activation]())

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            
            prev_feature = hidden_feature

        layers.append(
            nn.Linear(in_features=prev_feature, out_features=out_features)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Return logits from the configured sequential network."""
        return self.net(x)



class ModelTwo(nn.Module):
    """Three-hidden-layer ReLU classifier for tabular features."""

    def __init__(self, in_features, hidden_features, out_features):
        """Build three equal-width hidden layers and an output layer."""
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(in_features=hidden_features, out_features=out_features)
        
    def forward(self, x):
        """Apply the hidden layers and return binary logits."""
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.act3(x)
        x = self.layer4(x)
        return x


class ModelBatchNorm(nn.Module):
    """Three-hidden-layer classifier with batch normalization."""

    def __init__(self, in_features, hidden_features, out_features):
        """Build normalized hidden layers and the output layer."""
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.batch_norm1 = nn.BatchNorm1d(num_features=hidden_features)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.batch_norm2 = nn.BatchNorm1d(num_features=hidden_features)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.batch_norm3 = nn.BatchNorm1d(num_features=hidden_features)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(in_features=hidden_features, out_features=out_features)
        
    def forward(self, x):
        """Apply normalized hidden layers and return binary logits."""
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.batch_norm3(x)
        x = self.act3(x)
        x = self.layer4(x)
        return x
