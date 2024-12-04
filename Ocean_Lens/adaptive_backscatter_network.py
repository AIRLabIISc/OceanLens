import torch
import torch.nn as nn


class BackscatterNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer for backscatter and residual
        self.backscatter_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        self.residual_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        
        # Additional convolutional layers
        #self.backscatter_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        #self.residual_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        
        #self.backscatter_conv3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        #self.residual_conv3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        
        #self.backscatter_conv4 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        #self.residual_conv4 = nn.Conv2d(3, 3, 3, padding=1, bias=False)

        # Fifth convolutional layer (final layer for both paths)
        #self.backscatter_conv5 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        #self.residual_conv5 = nn.Conv2d(3, 3, 3, padding=1, bias=False)

        # Initialize all layers
        nn.init.uniform_(self.backscatter_conv1.weight, 0, 5)
        #nn.init.uniform_(self.backscatter_conv2.weight, 0, 5)
        #nn.init.uniform_(self.backscatter_conv3.weight, 0, 5)
        #nn.init.uniform_(self.backscatter_conv4.weight, 0, 5)
        #nn.init.uniform_(self.backscatter_conv5.weight, 0, 5)
        nn.init.uniform_(self.residual_conv1.weight, 0, 5)
        #nn.init.uniform_(self.residual_conv2.weight, 0, 5)
        #nn.init.uniform_(self.residual_conv3.weight, 0, 5)
        #nn.init.uniform_(self.residual_conv4.weight, 0, 5)
        #nn.init.uniform_(self.residual_conv5.weight, 0, 5)

        # Learned parameters
        self.I_B_infinity = nn.Parameter(torch.rand(3, 1, 1))
        self.I_B_prime = nn.Parameter(torch.rand(3, 1, 1))

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, image, depth):
        # First set of convolutions
        b1_conv1 = self.relu(self.backscatter_conv1(depth))
        b2_conv1 = self.relu(self.residual_conv1(depth))

        # Second set of convolutions
        #b1_conv2 = self.relu(self.backscatter_conv2(beta_b_conv1))
        #b2_conv2 = self.relu(self.residual_conv2(beta_d_conv1))

        # Third set of convolutions
        #b1_conv3 = self.relu(self.backscatter_conv3(beta_b_conv2))
        #b2_conv3 = self.relu(self.residual_conv3(beta_d_conv2))

        # Fourth set of convolutions
        #b1_conv4 = self.relu(self.backscatter_conv4(beta_b_conv3))
        #b2_conv4 = self.relu(self.residual_conv4(beta_d_conv3))

        # Fifth set of convolutions (final layer)
        #b1_conv5 = self.relu(self.backscatter_conv5(beta_b_conv4))
        #b2_conv5 = self.relu(self.residual_conv5(beta_d_conv4))

        # Backscatter calculation
        # change conv layer accordingly i.e to conv1 or conv2 or conv3 or conv4 or conv5
        I_Bc = self.I_B_infinity * (1 - torch.exp(-b1_conv1)) + self.I_B_prime * torch.exp(-b2_conv1) 
        backscatter_comp = self.sigmoid(I_Bc)

        # Masking backscatter and calculating the direct component
        backscatter_corr = backscatter_comp * (depth > 0.).repeat(1, 3, 1, 1)
        I_D = image - backscatter_corr

        return I_D, backscatter_comp

