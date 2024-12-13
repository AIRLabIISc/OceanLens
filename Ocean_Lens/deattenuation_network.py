import torch
import torch.nn as nn


class DeattenuateNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Original attenuation convolution layer
        self.deattenuation_conv = nn.Conv2d(1, 6, 1, bias=False)
        nn.init.uniform_(self.deattenuation_conv.weight, 0, 5)
        
        # New convolution layers
        self.conv1 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)
        #self.conv3 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)  # Third convolution layer
        #self.conv4 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1)  # Fourth convolution layer
        #self.conv5 = nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1)   # Fifth convolution layer

        self.a_f = nn.Parameter(torch.rand(6, 1, 1))
        self.relu = nn.ReLU()

        
        self.whitebalance = nn.Parameter(torch.rand(1, 1, 1))
        nn.init.constant_(self.whitebalance, 1)
        
        self.output_act = nn.Sigmoid()

    def forward(self, I_D, depth):
        # Pass depth through attenuation convolution layer
        deattn_conv = torch.exp(-self.relu(self.deattenuation_conv(depth)))
        
        # Pass through additional convolution layers
        deattn_conv = self.relu(self.conv1(deattn_conv))
        #deattn_conv = self.relu(self.conv2(deattn_conv))
        #deattn_conv = self.relu(self.conv3(deattn_conv))  # Third convolution layer
        #deattn_conv = self.relu(self.conv4(deattn_conv))  # Fourth convolution layer
        #deattn_conv = self.relu(self.conv5(deattn_conv))  # Fifth convolution layer
        
        # Calculate b_d for attenuation
        b_d = torch.stack(tuple(
            torch.sum(deattn_conv[:, i:i + 2, :, :] * self.relu(self.a_f[i:i + 2]), dim=1) for i in
            range(0, 6, 2)), dim=1)
        
        # Calculate b_d for attenuation -- changing number of exponentials
        #b_d = torch.stack(tuple(
         #   torch.sum(attn_conv[:, i:i + 3, :, :] * self.relu(self.a_f[i:i +3 ]), dim=1) for i in
         #   range(0, 6, 2)), dim=1) #3 exponentials
        

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



