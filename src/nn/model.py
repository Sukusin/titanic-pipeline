import torch
import torch.nn as nn

class ModelOne(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x
    

class ModelTwo(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(in_features=hidden_features, out_features=out_features)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.act3(x)
        x = self.layer4(x)
        return x


class ModelBatchNorm(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
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