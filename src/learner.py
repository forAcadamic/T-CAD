import torch
import torch.nn as nn
from torch.nn import functional as F

class Learner(nn.Module):
    def __init__(self, input_dim=1024, drop_p=0.0):
        super(Learner, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.drop_p = 0.6
        self.weight_init()
        self.vars = nn.ParameterList()

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        x = self.classifier(x)
        return x

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

class Pathleaner(nn.Module):
    def __init__(self, input_dim=31*18, drop_p=0.0):
        super(Pathleaner, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def weight_init(self):
        for layer in self.fc:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.fc(x)
