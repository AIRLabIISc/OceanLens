#3 convolution layer

class BackscatterNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer for backscatter and residual
        self.backscatter_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        self.residual_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        
        # Second convolutional layer
        self.backscatter_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)

        # Third convolutional layer (final layer for both paths)
        self.backscatter_conv3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)

        # Initialize all layers
        nn.init.uniform_(self.backscatter_conv1.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv2.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv3.weight, 0, 5)
        nn.init.uniform_(self.residual_conv1.weight, 0, 5)
        nn.init.uniform_(self.residual_conv2.weight, 0, 5)
        nn.init.uniform_(self.residual_conv3.weight, 0, 5)

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
        b1_conv2 = self.relu(self.backscatter_conv2(b1_conv1))
        b2_conv2 = self.relu(self.residual_conv2(b2_conv1))

        # Third set of convolutions (final layer)
        b1_conv3 = self.relu(self.backscatter_conv3(b1_conv2))
        b2_conv3 = self.relu(self.residual_conv3(b2_conv2))

        # Backscatter calculation
        I_Bc = self.I_B_infinity * (1 - torch.exp(-b1_conv3)) + self.I_B_prime * torch.exp(-b2_conv3)
        backscatter_comp = self.sigmoid(I_Bc)

        # Masking backscatter and calculating the direct component
        backscatter_corr = backscatter_comp * (depth > 0.).repeat(1, 3, 1, 1)
        I_D = image - backscatter_corr

        return I_D, backscatter_comp

class DeattenuateNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Original attenuation convolution layer
        self.attenuation_conv = nn.Conv2d(1, 6, 1, bias=False)
        nn.init.uniform_(self.attenuation_conv.weight, 0, 5)
        
        # New convolution layers
        self.conv1 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)  
        self.conv3 = nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1)   # Third convolution layer

        self.a_f = nn.Parameter(torch.rand(6, 1, 1))
        self.relu = nn.ReLU()

        # White balance parameter
        self.whitebalance = nn.Parameter(torch.rand(1, 1, 1))
        nn.init.constant_(self.whitebalance, 1)
        
        self.output_act = nn.Sigmoid()

    def forward(self, I_D, depth):
        # Pass depth through attenuation convolution layer
        deattn_conv = torch.exp(-self.relu(self.attenuation_conv(depth)))
        
        # Pass through additional convolution layers
        deattn_conv = self.relu(self.conv1(deattn_conv))
        deattn_conv = self.relu(self.conv2(deattn_conv))
        deattn_conv = self.relu(self.conv3(deattn_conv))  # Third convolution layer
        
        # Calculate b_d for attenuation
        b_d = torch.stack(tuple(
            torch.sum(deattn_conv[:, i:i + 2, :, :] * self.relu(self.a_f[i:i + 2]), dim=1) for i in
            range(0, 6, 2)), dim=1)

        # make it i+3 to change no. of exponents
      
        # Calculate correction_factor
        correction_factor = torch.exp(torch.clamp(b_d * depth, 0, float(torch.log(torch.tensor([3.])))))
        
        # Handle correction_factor for depth 0 and greater than 0 cases
        correction_factor_depth = correction_factor * ((depth == 0.) / correction_factor + (depth > 0.))
        
        # Final output 
        I = correction_factor_depth * I_D * self.whitebalance
        
        # Handle NaN values in I
        nanmask = torch.isnan(I)
        if torch.any(nanmask):
            print("Warning! NaN values in I")
            I[nanmask] = 0
            
        return correction_factor_depth, I



# 4 convolution layers

class BackscatterNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer for backscatter and residual
        self.backscatter_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        self.residual_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        
        # Additional convolutional layers
        self.backscatter_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        
        self.backscatter_conv3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)

        # Fourth convolutional layer (final layer for both paths)
        self.backscatter_conv4 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv4 = nn.Conv2d(3, 3, 3, padding=1, bias=False)

        # Initialize all layers
        nn.init.uniform_(self.backscatter_conv1.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv2.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv3.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv4.weight, 0, 5)
        nn.init.uniform_(self.residual_conv1.weight, 0, 5)
        nn.init.uniform_(self.residual_conv2.weight, 0, 5)
        nn.init.uniform_(self.residual_conv3.weight, 0, 5)
        nn.init.uniform_(self.residual_conv4.weight, 0, 5)

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
        b1_conv2 = self.relu(self.backscatter_conv2(b1_conv1))
        b2_conv2 = self.relu(self.residual_conv2(b2_conv1))

        # Third set of convolutions
        b1_conv3 = self.relu(self.backscatter_conv3(b1_conv2))
        b2_conv3 = self.relu(self.residual_conv3(b2_conv2))

        # Fourth set of convolutions (final layer)
        b1_conv4 = self.relu(self.backscatter_conv4(b1_conv3))
        b2_conv4 = self.relu(self.residual_conv4(b2_conv3))

        # Backscatter calculation
        I_Bc = self.I_B_infinity * (1 - torch.exp(-b1_conv4)) + self.I_B_prime * torch.exp(-b2_conv4)
        backscatter_comp = self.sigmoid(I_Bc)

        # Masking backscatter and calculating the direct component
        backscatter_corr = backscatter_comp * (depth > 0.).repeat(1, 3, 1, 1)
        I_D = image - backscatter_corr

        return I_D, backscatter_comp



class DeattenuateNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Original attenuation convolution layer
        self.attenuation_conv = nn.Conv2d(1, 6, 1, bias=False)
        nn.init.uniform_(self.attenuation_conv.weight, 0, 5)
        
        # New convolution layers
        self.conv1 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)  # Third convolution layer
        self.conv4 = nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1)   # Fourth convolution layer

        self.a_f = nn.Parameter(torch.rand(6, 1, 1))
        self.relu = nn.ReLU()

        # White balance parameter
        self.whitebalance = nn.Parameter(torch.rand(1, 1, 1))
        nn.init.constant_(self.whitebalance, 1)
        
        self.output_act = nn.Sigmoid()

    def forward(self, I_D, depth):
        # Pass depth through attenuation convolution layer
        deattn_conv = torch.exp(-self.relu(self.attenuation_conv(depth)))
        
        # Pass through additional convolution layers
        deattn_conv = self.relu(self.conv1(deattn_conv))
        deattn_conv = self.relu(self.conv2(deattn_conv))
        deattn_conv = self.relu(self.conv3(deattn_conv))  # Third convolution layer
        deattn_conv = self.relu(self.conv4(deattn_conv))  # Fourth convolution layer
        
        # Calculate b_d for attenuation
        b_d = torch.stack(tuple(
            torch.sum(deattn_conv[:, i:i + 2, :, :] * self.relu(self.a_f[i:i + 2]), dim=1) for i in
            range(0, 6, 2)), dim=1)
        
        # Calculate correction_factor
        correction_factor = torch.exp(torch.clamp(b_d * depth, 0, float(torch.log(torch.tensor([3.])))))
        
        # Handle correction_factor for depth 0 and greater than 0 cases
        correction_factor_depth = correction_factor * ((depth == 0.) / correction_factor + (depth > 0.))
        
        # Final output I
        I = correction_factor_depth * I_D * self.whitebalance
        
        # Handle NaN values in I
        nanmask = torch.isnan(I)
        if torch.any(nanmask):
            print("Warning! NaN values in I")
            I[nanmask] = 0
            
        return correction_factor_depth, I

# 5 layers
class BackscatterNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer for backscatter and residual
        self.backscatter_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        self.residual_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        
        # Additional convolutional layers
        self.backscatter_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        
        self.backscatter_conv3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        
        self.backscatter_conv4 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv4 = nn.Conv2d(3, 3, 3, padding=1, bias=False)

        # Fifth convolutional layer (final layer for both paths)
        self.backscatter_conv5 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv5 = nn.Conv2d(3, 3, 3, padding=1, bias=False)

        # Initialize all layers
        nn.init.uniform_(self.backscatter_conv1.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv2.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv3.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv4.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv5.weight, 0, 5)
        nn.init.uniform_(self.residual_conv1.weight, 0, 5)
        nn.init.uniform_(self.residual_conv2.weight, 0, 5)
        nn.init.uniform_(self.residual_conv3.weight, 0, 5)
        nn.init.uniform_(self.residual_conv4.weight, 0, 5)
        nn.init.uniform_(self.residual_conv5.weight, 0, 5)

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
        b1_conv2 = self.relu(self.backscatter_conv2(b1_conv1))
        b2_conv2 = self.relu(self.residual_conv2(b2_conv1))

        # Third set of convolutions
        b1_conv3 = self.relu(self.backscatter_conv3(b1_conv2))
        b2_conv3 = self.relu(self.residual_conv3(b2_conv2))

        # Fourth set of convolutions
        b1_conv4 = self.relu(self.backscatter_conv4(b1_conv3))
        b2_conv4 = self.relu(self.residual_conv4(b2_conv3))

        # Fifth set of convolutions (final layer)
        b1_conv5 = self.relu(self.backscatter_conv5(b1_conv4))
        b2_conv5 = self.relu(self.residual_conv5(b2_conv4))

        # Backscatter calculation
        I_Bc = self.I_B_infinity * (1 - torch.exp(-b1_conv5)) + self.I_B_prime * torch.exp(-b2_conv5)
        backscatter_comp = self.sigmoid(I_Bc)

        # Masking backscatter and calculating the direct component
        backscatter_corr = backscatter_comp * (depth > 0.).repeat(1, 3, 1, 1)
        I_D = image - backscatter_corr

        return I_D, backscatter_comp

class DeattenuateNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Original attenuation convolution layer
        self.attenuation_conv = nn.Conv2d(1, 6, 1, bias=False)
        nn.init.uniform_(self.attenuation_conv.weight, 0, 5)
        
        # New convolution layers
        self.conv1 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)  # Third convolution layer
        self.conv4 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)  # Fourth convolution layer
        self.conv5 = nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1)   # Fifth convolution layer

        self.a_f = nn.Parameter(torch.rand(6, 1, 1))
        self.relu = nn.ReLU()

        # White balance parameter
        self.whitebalance = nn.Parameter(torch.rand(1, 1, 1))
        nn.init.constant_(self.whitebalance, 1)
        
        self.output_act = nn.Sigmoid()

    def forward(self, I_D, depth):
        # Pass depth through attenuation convolution layer
        deattn_conv = torch.exp(-self.relu(self.attenuation_conv(depth)))
        
        # Pass through additional convolution layers
        deattn_conv = self.relu(self.conv1(deattn_conv))
        deattn_conv = self.relu(self.conv2(deattn_conv))
        deattn_conv = self.relu(self.conv3(deattn_conv))  # Third convolution layer
        deattn_conv = self.relu(self.conv4(deattn_conv))  # Fourth convolution layer
        deattn_conv = self.relu(self.conv5(deattn_conv))  # Fifth convolution layer
        
        # Calculate b_d for attenuation
        b_d = torch.stack(tuple(
            torch.sum(deattn_conv[:, i:i + 2, :, :] * self.relu(self.a_f[i:i + 2]), dim=1) for i in
            range(0, 6, 2)), dim=1)
        
        # Calculate correction_factor
        correction_factor = torch.exp(torch.clamp(b_d * depth, 0, float(torch.log(torch.tensor([3.])))))
        
        # Handle correction_factor for depth 0 and greater than 0 cases
        correction_factor_depth = correction_factor * ((depth == 0.) / correction_factor + (depth > 0.))
        
        # Final output I
        I = correction_factor_depth * I_D * self.whitebalance
        
        # Handle NaN values in I
        nanmask = torch.isnan(I)
        if torch.any(nanmask):
            print("Warning! NaN values in I")
            I[nanmask] = 0
            
        return correction_factor_depth, I


# 6 layers
import torch
import torch.nn as nn

class BackscatterNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer for backscatter and residual
        self.backscatter_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        self.residual_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        
        # Additional convolutional layers
        self.backscatter_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        
        self.backscatter_conv3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv3 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        
        self.backscatter_conv4 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv4 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        
        self.backscatter_conv5 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv5 = nn.Conv2d(3, 3, 3, padding=1, bias=False)

        # New sixth convolutional layer
        self.backscatter_conv6 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.residual_conv6 = nn.Conv2d(3, 3, 3, padding=1, bias=False)

        # Initialize all layers
        nn.init.uniform_(self.backscatter_conv1.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv2.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv3.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv4.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv5.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv6.weight, 0, 5) 
        nn.init.uniform_(self.residual_conv1.weight, 0, 5)
        nn.init.uniform_(self.residual_conv2.weight, 0, 5)
        nn.init.uniform_(self.residual_conv3.weight, 0, 5)
        nn.init.uniform_(self.residual_conv4.weight, 0, 5)
        nn.init.uniform_(self.residual_conv5.weight, 0, 5)
        nn.init.uniform_(self.residual_conv6.weight, 0, 5)  

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
        b1_conv2 = self.relu(self.backscatter_conv2(b1_conv1))
        b2_conv2 = self.relu(self.residual_conv2(b2_conv1))

        # Third set of convolutions
        b1_conv3 = self.relu(self.backscatter_conv3(b1_conv2))
        b2_conv3 = self.relu(self.residual_conv3(b2_conv2))

        # Fourth set of convolutions
        b1_conv4 = self.relu(self.backscatter_conv4(b1_conv3))
        b2_conv4 = self.relu(self.residual_conv4(b2_conv3))

        # Fifth set of convolutions
        b1_conv5 = self.relu(self.backscatter_conv5(b1_conv4))
        b2_conv5 = self.relu(self.residual_conv5(b2_conv4))

        # New sixth set of convolutions
        b1_conv6 = self.relu(self.backscatter_conv6(b1_conv5))
        b2_conv6 = self.relu(self.residual_conv6(b2_conv5))

        # Backscatter calculation
        I_Bc = self.I_B_infinity * (1 - torch.exp(-b1_conv6)) + self.I_B_prime * torch.exp(-b2_conv6)
        backscatter_comp = self.sigmoid(I_Bc)

        # Masking backscatter and calculating the direct component
        backscatter_corr = backscatter_comp * (depth > 0.).repeat(1, 3, 1, 1)
        I_D = image - backscatter_corr

        return I_D, backscatter_comp

class DeattenuateNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Original attenuation convolution layer
        self.attenuation_conv = nn.Conv2d(1, 6, 1, bias=False)
        nn.init.uniform_(self.attenuation_conv.weight, 0, 5)
        
        # New convolution layers
        self.conv1 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)  # Third convolution layer
        self.conv4 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)  # Fourth convolution layer
        self.conv5 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)  # Fifth convolution layer
        self.conv6 = nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1)   # Sixth convolution layer

        self.a_f = nn.Parameter(torch.rand(6, 1, 1))
        self.relu = nn.ReLU()

        # White balance parameter
        self.whitebalance = nn.Parameter(torch.rand(1, 1, 1))
        nn.init.constant_(self.whitebalance, 1)
        
        self.output_act = nn.Sigmoid()

    def forward(self, I_D, depth):
        # Pass depth through attenuation convolution layer
        deattn_conv = torch.exp(-self.relu(self.attenuation_conv(depth)))
        
        # Pass through additional convolution layers
        deattn_conv = self.relu(self.conv1(deattn_conv))
        deattn_conv = self.relu(self.conv2(deattn_conv))
        deattn_conv = self.relu(self.conv3(deattn_conv))  # Third convolution layer
        deattn_conv = self.relu(self.conv4(deattn_conv))  # Fourth convolution layer
        deattn_conv = self.relu(self.conv5(deattn_conv))  # Fifth convolution layer
        deattn_conv = self.relu(self.conv6(deattn_conv))  # Sixth convolution layer
        
        # Calculate b_d for attenuation
        b_d = torch.stack(tuple(
            torch.sum(deattn_conv[:, i:i + 2, :, :] * self.relu(self.a_f[i:i + 2]), dim=1) for i in
            range(0, 6, 2)), dim=1)
        
        # Calculate correction_factor
        correction_factor = torch.exp(torch.clamp(b_d * depth, 0, float(torch.log(torch.tensor([3.])))))
        
        # Handle correction_factor for depth 0 and greater than 0 cases
        correction_factor_depth = correction_factor * ((depth == 0.) / correction_factor + (depth > 0.))
        
        # Final output I
        I = correction_factor_depth * I_D * self.whitebalance
        
        # Handle NaN values in I
        nanmask = torch.isnan(I)
        if torch.any(nanmask):
            print("Warning! NaN values in I")
            I[nanmask] = 0
            
        return correction_factor_depth, I


