import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, in_features, hidden_feature, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_features, out_features=hidden_feature)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(in_features=hidden_feature, out_features=out_features)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x
    

if __name__ == "__main__":
    net = BaselineModel(5, 16, 1)
    print(net)
