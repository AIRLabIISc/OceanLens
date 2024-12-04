import torch
import torch.nn as nn



class BackscatterLoss(nn.Module):
    def __init__(self, cost_ratio=1000.0, delta=1):
        super(BackscatterLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio
        self.delta = delta

    def adaptive_huber(self, prediction, target):
        abs_error = torch.abs(prediction - target)
        quadratic = torch.where(abs_error <= self.delta, 0.5 * abs_error ** 2, self.delta * (abs_error - 0.5 * self.delta))
        return quadratic

    def forward(self, I_D):
        # Positive loss (ReLU applied to I_D)
        pos = self.l1(self.relu(I_D), torch.zeros_like(I_D))
        
        # Negative loss (ReLU applied to -I_D using Adaptive Huber Loss)
        neg = self.adaptive_huber(self.relu(-I_D), torch.zeros_like(I_D))
        neg_mean = torch.mean(neg)
        
        # Backscatter loss combining positive and negative parts
        bs_loss = self.cost_ratio * neg_mean + pos
        
        return bs_loss



