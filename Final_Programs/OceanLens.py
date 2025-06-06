# Copyright 2024 Rajini Makam,Indian Institute of Science
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License, version 3, as published by the Free Software Foundation.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.


import os
import argparse
import gzip
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_laplace

from time import time

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.utils import save_image
# from transformers import pipeline
import kornia.morphology as morph
from PIL import Image
try:
    from tqdm import trange , tqdm
except:
    trange = range

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import tqdm
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import logging 
 # type: ignore
import cv2 
import datetime
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_laplace
import cv2
from scipy import ndimage
from PIL import Image
import numpy as np
import math

# Device definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag) 
    return mag

def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size

    # Print k1 and k2 for debugging
    #print("k1:", k1, "type:", type(k1))
    #print("k2:", k2, "type:", type(k2))
    
    # weight
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    #print("Initial blocksize_y:", blocksize_y, "type:", type(blocksize_y))
    #print("Initial blocksize_x:", blocksize_x, "type:", type(blocksize_x))

    # Ensure blocksize_y, blocksize_x, k1, and k2 are integers
    blocksize_y = int(blocksize_y)
    blocksize_x = int(blocksize_x)
    k1 = int(k1)
    k2 = int(k2)

    # Print converted values and types for debugging
    #print("Converted blocksize_y:", blocksize_y, "type:", type(blocksize_y))
    #print("Converted blocksize_x:", blocksize_x, "type:", type(blocksize_x))
    #print("Converted k1:", k1, "type:", type(k1))
    #print("Converted k2:", k2, "type:", type(k2))

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y*k2, :blocksize_x*k1]
    # Print the shape of x for debugging
    #print("Shape of x:", x.shape)
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            # bound checks, can't do log(0)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val

def _uism(x):
    #print("Shape of input image:", x.shape)
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    #print("R channel shape:", R.shape)
    #print("G channel shape:", G.shape)
    #print("B channel shape:", B.shape)

    # Apply Sobel filter
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    #print("Sobel R shape:", Rs.shape)
    #print("Sobel G shape:", Gs.shape)
    #print("Sobel B shape:", Bs.shape)

    # Multiply edges by channels
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    #print("R_edge_map shape:", R_edge_map.shape)
    #print("G_edge_map shape:", G_edge_map.shape)
    #print("B_edge_map shape:", B_edge_map.shape)

    # Get EME for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    #print("r_eme:", r_eme)
    #print("g_eme:", g_eme)
    #print("b_eme:", b_eme)

    # Coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def plip_g(x,mu=1026.0):
    return mu-x

def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k*((g1-g2)/(k-g2))

def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1+g2-((g1*g2)/(gamma))

def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g/gamma) ), c) )

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    #return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))

def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));

def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)

def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/abstract/document/5609219
    """
    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    #print("k1:", k1, "type:", type(k1))
    #print("k2:", k2, "type:", type(k2))
    k1 = int(k1)
    k2 = int(k2)
    
    # weight
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    #print("Initial blocksize_y:", blocksize_y, "type:", type(blocksize_y))
    #print("Initial blocksize_x:", blocksize_x, "type:", type(blocksize_x))
    blocksize_y = int(blocksize_y)
    blocksize_x = int(blocksize_x)
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y*k2, :blocksize_x*k1]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w*val

def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    ### UCIQE: https://ieeexplore.ieee.org/abstract/document/7300447
    #c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### UIQM https://ieeexplore.ieee.org/abstract/document/7305804
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm,uicm,uism,uiconm

def getUCIQE(img):
    img_BGR = cv2.imread(img)
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB) 
    img_LAB = np.array(img_LAB,dtype=np.float64)
    # Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576 according to paper.
    coe_Metric = [0.4680, 0.2745, 0.2576]
    
    img_lum = img_LAB[:,:,0]/255.0
    img_a = img_LAB[:,:,1]/255.0
    img_b = img_LAB[:,:,2]/255.0

    # item-1
    chroma = np.sqrt(np.square(img_a)+np.square(img_b))
    sigma_c = np.std(chroma)

    # item-2
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum)
    top_index = sorted_index[int(len(img_lum)*0.99)]
    bottom_index = sorted_index[int(len(img_lum)*0.01)]
    con_lum = img_lum[top_index] - img_lum[bottom_index]

    # item-3
    chroma = chroma.flatten()
    sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum!=0)
    avg_sat = np.mean(sat)

    uciqe = sigma_c*coe_Metric[0] + con_lum*coe_Metric[1] + avg_sat*coe_Metric[2]
    return uciqe

def measure_UIQMs(dir_name, file_ext=None):
    """
      # measured in RGB
      Assumes:
        * dir_name contain generated images 
        * to evaluate on all images: file_ext = None 
        * to evaluate images that ends with "_SESR.png" or "_En.png"  
            * use file_ext = "_SESR.png" or "_En.png" 
    """
    paths = sorted(glob(join(dir_name, "*.*")))
    if file_ext:
        paths = [p for p in paths if p.endswith(file_ext)]
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize((640,640))
        uqims.append(getUIQM(np.array(im)))
    return np.array(uqims)


class paired_rgb_depth_dataset(Dataset):
    def __init__(self, image_path, depth_path, openni_depth, mask_max_depth, image_height, image_width):
        self.image_dir = image_path
        self.depth_dir = depth_path
        self.image_files = sorted(os.listdir(image_path))
        self.depth_files = sorted(os.listdir(depth_path))
        self.openni_depth = openni_depth
        self.mask_max_depth = mask_max_depth
        self.crop = (0, 0, image_height, image_width)
        self.depth_perc = 0.0001
        self.kernel = torch.ones(3, 3).to(device)
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.crop[2], self.crop[3]), transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.PILToTensor(),
        ])
        assert len(self.image_files) == len(self.depth_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        fname = self.image_files[index]
        image = Image.open(os.path.join(self.image_dir, fname))
        depth_fname = self.depth_files[index]
        depth = Image.open(os.path.join(self.depth_dir, depth_fname))
        if depth.mode != 'L':
            depth = depth.convert('L')
        depth_transformed: torch.Tensor = self.image_transforms(depth).float().to(device)
        if self.openni_depth:
            depth_transformed = depth_transformed / 1000.
        if self.mask_max_depth:
            depth_transformed[depth_transformed == 0.] = depth_transformed.max()
        low, high = torch.nanquantile(depth_transformed, self.depth_perc), torch.nanquantile(depth_transformed,
                                                                                             1. - self.depth_perc)
        depth_transformed[(depth_transformed < low) | (depth_transformed > high)] = 0.
        depth_transformed = torch.squeeze(morph.closing(torch.unsqueeze(depth_transformed, dim=0), self.kernel), dim=0)
        left_transformed: torch.Tensor = self.image_transforms(image).to(device) / 255.
        return left_transformed, depth_transformed, [fname]


class BackscatterNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer for backscatter and residual
        self.backscatter_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        self.residual_conv1 = nn.Conv2d(1, 3, 1, bias=False)
        
        # Additional convolutional layers as per the comment
        self.backscatter_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)  # New backscatter conv layer
        self.residual_conv2 = nn.Conv2d(3, 3, 3, padding=1, bias=False)  # New residual conv layer

        nn.init.uniform_(self.backscatter_conv1.weight, 0, 5)
        nn.init.uniform_(self.backscatter_conv2.weight, 0, 5)
        nn.init.uniform_(self.residual_conv1.weight, 0, 5)
        nn.init.uniform_(self.residual_conv2.weight, 0, 5)

        self.B_inf = nn.Parameter(torch.rand(3, 1, 1))
        self.J_prime = nn.Parameter(torch.rand(3, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, image, depth):
        # First set of convolutions
        beta_b_conv1 = self.relu(self.backscatter_conv1(depth))
        beta_d_conv1 = self.relu(self.residual_conv1(depth))

        # Second set of convolutions (feature-feature equation)
        beta_b_conv2 = self.relu(self.backscatter_conv2(beta_b_conv1))
        beta_d_conv2 = self.relu(self.residual_conv2(beta_d_conv1))

        # Backscatter calculation
        Bc = self.B_inf * (1 - torch.exp(-beta_b_conv2)) + self.J_prime * torch.exp(-beta_d_conv2)
        backscatter = self.sigmoid(Bc)

        # Masking backscatter and calculating the direct component
        backscatter_masked = backscatter * (depth > 0.).repeat(1, 3, 1, 1)
        direct = image - backscatter_masked

        return direct, backscatter



class DeattenuateNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Original attenuation convolution layer
        self.attenuation_conv = nn.Conv2d(1, 6, 1, bias=False)
        nn.init.uniform_(self.attenuation_conv.weight, 0, 5)
        
        # New convolution layers
        self.conv1 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1)
        
        self.attenuation_coef = nn.Parameter(torch.rand(6, 1, 1))
        self.relu = nn.ReLU()

        # White balance parameter
        self.wb = nn.Parameter(torch.rand(1, 1, 1))
        nn.init.constant_(self.wb, 1)
        
        self.output_act = nn.Sigmoid()

    def forward(self, direct, depth):
        # Pass depth through attenuation convolution layer
        attn_conv = torch.exp(-self.relu(self.attenuation_conv(depth)))
        
        # Pass through additional convolution layers
        attn_conv = self.relu(self.conv1(attn_conv))
        attn_conv = self.relu(self.conv2(attn_conv))
        
        # Calculate beta_d for attenuation
        beta_d = torch.stack(tuple(
            torch.sum(attn_conv[:, i:i + 2, :, :] * self.relu(self.attenuation_coef[i:i + 2]), dim=1) for i in
            range(0, 6, 2)), dim=1)
        
        # Calculate f
        f = torch.exp(torch.clamp(beta_d * depth, 0, float(torch.log(torch.tensor([3.])))))
        
        # Handle f_masked for depth 0 and greater than 0 cases
        f_masked = f * ((depth == 0.) / f + (depth > 0.))
        
        # Final output J
        J = f_masked * direct * self.wb
        
        # Handle NaN values in J
        nanmask = torch.isnan(J)
        if torch.any(nanmask):
            print("Warning! NaN values in J")
            J[nanmask] = 0
            
        return f_masked, J



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

    def forward(self, direct):
        # Positive loss (ReLU applied to direct)
        pos = self.l1(self.relu(direct), torch.zeros_like(direct))
        
        # Negative loss (ReLU applied to -direct using Adaptive Huber Loss)
        neg = self.adaptive_huber(self.relu(-direct), torch.zeros_like(direct))
        neg_mean = torch.mean(neg)
        
        # Backscatter loss combining positive and negative parts
        bs_loss = self.cost_ratio * neg_mean + pos
        
        return bs_loss


# class BackscatterLoss(nn.Module):
#     def __init__(self, cost_ratio=1000.):
#         super().__init__()
#         self.l1 = nn.L1Loss()
#         self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
#         self.mse = nn.MSELoss()
#         self.relu = nn.ReLU()
#         self.cost_ratio = cost_ratio

#     def forward(self, direct):
#         pos = self.l1(self.relu(direct), torch.zeros_like(direct))
#         neg = self.smooth_l1(self.relu(-direct), torch.zeros_like(direct))
#         bs_loss = self.cost_ratio * neg + pos
#         return bs_loss


class DeattenuateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_intensity = 0.5

    def laplacian_of_gaussian(self, img):
            # Ensure 'img' is a NumPy array
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()

            # If the image is still 4D after squeezing, handle it
            if img.ndim == 4:
                # Typically, we take the first image in the batch and the first channel
                img = img[0, 0]  # This will give you the height x width part, reducing to 2D

            # If the image is 3D (like RGB), convert it to grayscale
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.mean(img, axis=0)  # Convert to grayscale by averaging across channels

            # Ensure the result is 2D
            img = np.squeeze(img)

            # Define Laplacian of Gaussian (LoG) filter
            gaussian_kernel_3x3 = np.array([[1, 2, 1],
                                            [2, 4, 2],
                                            [1, 2, 1]], dtype=np.float32) / 16
            
            # Apply Gaussian filter
            gauss_img = cv2.filter2D(img, -1, gaussian_kernel_3x3)
            
            # Define Laplacian filter
            laplacian_kernel = np.array([[0, -1, 0],
                                        [-1, 4, -1],
                                        [0, -1, 0]], dtype=np.float32)
            
            # Apply Laplacian filter
            laplacian_img = cv2.filter2D(gauss_img, -1, laplacian_kernel)

            return laplacian_img

    def log_loss(self, Y_pred, Y_true):
            # Define Gaussian kernel
            gaussian_kernel_3x3 = np.array([[1, 2, 1],
                                            [2, 4, 2],
                                            [1, 2, 1]], dtype=np.float32) / 16
            
            # Initialize arrays for filtered results
            gauss_pred = np.zeros_like(Y_pred.cpu().numpy())
            gauss_true = np.zeros_like(Y_true.cpu().numpy())

            # Ensure inputs are numpy arrays
            Y_pred = Y_pred.cpu().numpy()
            Y_true = Y_true.cpu().numpy()

            if Y_pred.ndim == 4:
                for i in range(Y_pred.shape[0]):
                    img_pred = Y_pred[i]
                    img_true = Y_true[i]
                    
                    # Handle 4D tensors by reducing to 2D if necessary
                    if img_pred.ndim == 4:
                        img_pred = img_pred[0, 0]  # Reduce to 2D (height x width)

                    if img_true.ndim == 4:
                        img_true = img_true[0, 0]  # Reduce to 2D (height x width)
                        
                    # Convert to grayscale if RGB
                    if img_pred.ndim == 3 and img_pred.shape[0] == 3:
                        img_pred = np.mean(img_pred, axis=0)
                        
                    if img_true.ndim == 3 and img_true.shape[0] == 3:
                        img_true = np.mean(img_true, axis=0)
                    
                    # Apply Gaussian filter to each image
                    if img_pred.ndim == 3:  # RGB image case
                        gauss_pred_channels = []
                        for channel in range(img_pred.shape[2]):
                            gauss_pred_channel = cv2.filter2D(img_pred[:, :, channel], -1, gaussian_kernel_3x3)
                            gauss_pred_channels.append(gauss_pred_channel)
                        gauss_pred[i] = np.stack(gauss_pred_channels, axis=-1)  # Combine channels
                    else:
                        gauss_pred[i] = cv2.filter2D(img_pred, -1, gaussian_kernel_3x3)
                        
                    gauss_true[i] = cv2.filter2D(img_true, -1, gaussian_kernel_3x3)

            # Compute the loss
            laplacian_pred = self.laplacian_of_gaussian(Y_pred)  # Assuming laplacian_of_gaussian method exists
            laplacian_true = self.laplacian_of_gaussian(Y_true)
            loss = np.mean(np.abs(gauss_pred * laplacian_pred - gauss_true * laplacian_true))
            
            return loss
    
    def sobel_edge_loss(self, I_pred, I_true):
            sobel_x = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
            sobel_y = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])
        
        
            # Assuming 'img' is a 4D tensor in the form (batch_size, channels, height, width)
            I_pred = I_pred.squeeze().cpu().numpy()

            #    If the image is still 4D after squeezing, handle it
            if I_pred.ndim == 4:
                # Typically, we take the first image in the batch and the first channel
                I_pred = I_pred[0, 0]  # This will give you the height x width part, reducing to 2D

            #        If the image is 3D (like RGB), convert it to grayscale
            if I_pred.ndim == 3 and I_pred.shape[0] == 3:
                I_pred = np.mean(I_pred, axis=0)  # Convert to grayscale by averaging across channels

            # Ensure the result is 2D
            I_pred = np.squeeze(I_pred)

            #I_true = I_true.cpu().numpy()
            
            # Assuming 'img' is a 4D tensor in the form (batch_size, channels, height, width)
            I_true = I_true.squeeze().cpu().numpy()

            #    If the image is still 4D after squeezing, handle it
            if I_true.ndim == 4:
                # Typically, we take the first image in the batch and the first channel
                I_true = I_true[0, 0]  # This will give you the height x width part, reducing to 2D

            #        If the image is 3D (like RGB), convert it to grayscale
            if I_true.ndim == 3 and I_true.shape[0] == 3:
                I_true = np.mean(I_true, axis=0)  # Convert to grayscale by averaging across channels

            # Ensure the result is 2D
            I_true = np.squeeze(I_true)

            if I_pred.ndim == 4:
                I_pred = I_pred[0]
                I_true = I_true[0]
        
            loss_total = 0

            for c in range(I_pred.shape[0]):
                sobel_x_pred = cv2.filter2D(I_pred[c], -1, sobel_x)
                sobel_x_true = cv2.filter2D(I_true[c], -1, sobel_x)
                sobel_y_pred = cv2.filter2D(I_pred[c], -1, sobel_y)
                sobel_y_true = cv2.filter2D(I_true[c], -1, sobel_y)
            
                loss_x = np.abs(sobel_x_pred - sobel_x_true)
                loss_y = np.abs(sobel_y_pred - sobel_y_true)
            
                loss_total += np.mean(loss_x + loss_y)
        
            return loss_total / I_pred.shape[0]

    def dog_loss(self, Y_pred, Y_true):
            gaussian_kernel_3x3 = np.array([[1, 2, 1],
                                            [2, 4, 2],
                                            [1, 2, 1]]) / 16
            gaussian_kernel_5x5 = np.array([[1, 4, 6, 4, 1],
                                            [4, 16, 24, 16, 4],
                                            [6, 24, 36, 24, 6],
                                            [4, 16, 24, 16, 4],
                                            [1, 4, 6, 4, 1]]) / 256

            batch_size = Y_pred.shape[0]
            loss_total = 0

            for i in range(batch_size):
                img_pred = Y_pred[i].squeeze().cpu().numpy()
                img_true = Y_true[i].squeeze().cpu().numpy()
                # Assuming 'img' is a 4D tensor in the form (batch_size, channels, height, width)
                
                #    If the image is still 4D after squeezing, handle it
                if img_pred.ndim == 4:
                # Typically, we take the first image in the batch and the first channel
                    img_pred= img_pred[0, 0]  # This will give you the height x width part, reducing to 2D

                #        If the image is 3D (like RGB), convert it to grayscale
                if img_pred.ndim == 3 and img_pred.shape[0] == 3:
                    img_pred= np.mean(img_pred, axis=0)  # Convert to grayscale by averaging across channels

                # Ensure the result is 2D
                img_pred = np.squeeze(img_pred)

                # Assuming 'img' is a 4D tensor in the form (batch_size, channels, height, width)
                
                #    If the image is still 4D after squeezing, handle it
                if img_true.ndim == 4:
                # Typically, we take the first image in the batch and the first channel
                    img_true= img_true[0, 0]  # This will give you the height x width part, reducing to 2D

                #        If the image is 3D (like RGB), convert it to grayscale
                if img_true.ndim == 3 and img_true.shape[0] == 3:
                    img_true= np.mean(img_true, axis=0)  # Convert to grayscale by averaging across channels

                # Ensure the result is 2D
                img_true = np.squeeze(img_true)

                

                gauss_1_pred = np.array([cv2.filter2D(img_pred[c], -1, gaussian_kernel_3x3) for c in range(img_pred.shape[0])])
                gauss_1_true = np.array([cv2.filter2D(img_true[c], -1, gaussian_kernel_3x3) for c in range(img_true.shape[0])])
                gauss_2_pred = np.array([cv2.filter2D(img_pred[c], -1, gaussian_kernel_5x5) for c in range(img_pred.shape[0])])
                gauss_2_true = np.array([cv2.filter2D(img_true[c], -1, gaussian_kernel_5x5) for c in range(img_true.shape[0])])

                dog_pred = gauss_1_pred - gauss_2_pred
                dog_true = gauss_1_true - gauss_2_true

                loss_total += np.mean(np.abs(dog_pred - dog_true))

            return loss_total / batch_size

    def ssim_loss(self, Y_pred, Y_true):
            # Convert PyTorch tensors to NumPy arrays
            Y_pred_np = Y_pred.cpu().numpy()
            Y_true_np = Y_true.cpu().numpy()

            # Define the default window size (should be odd and less than or equal to the image dimensions)
            default_win_size = 7

            # Calculate SSIM for each image in the batch
            ssim_values = []
            for i in range(Y_pred_np.shape[0]):
                pred_image = Y_pred_np[i].transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
                true_image = Y_true_np[i].transpose(1, 2, 0)

                # Ensure that window size does not exceed image dimensions
                min_dim = min(pred_image.shape[0], pred_image.shape[1])
                win_size = min(default_win_size, min_dim)  # Adjust win_size if necessary
                
                # Handle dimensionality: Convert to grayscale if needed
                if pred_image.ndim == 3 and pred_image.shape[2] == 3:
                    pred_image = np.mean(pred_image, axis=2)  # Convert to grayscale
                if true_image.ndim == 3 and true_image.shape[2] == 3:
                    true_image = np.mean(true_image, axis=2)  # Convert to grayscale

                # Ensure the result is 2D
                pred_image = np.squeeze(pred_image)
                true_image = np.squeeze(true_image)

                # Compute SSIM
                ssim_value = ssim(pred_image, true_image, win_size=win_size, data_range=true_image.max() - true_image.min(), multichannel=False)
                ssim_values.append(ssim_value)

            return np.mean(ssim_values)

    def forward(self, direct, J):
        saturation_loss = (self.relu(-J) + self.relu(J - 1)).square().mean()
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        channel_spatial = torch.std(J, dim=[2, 3])
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        # Compute additional losses
        Y_pred = J.cpu().detach()
        Y_true = direct.cpu().detach()
        # log_loss_value = self.log_loss(Y_pred, Y_true)
        sobel_loss_value = self.sobel_edge_loss(Y_pred, Y_true)
        log_loss_value = self.log_loss(Y_pred, Y_true)
        # dog_loss_value = self.dog_loss(Y_pred, Y_true)
        # sobel_loss_value = self.sobel_edge_loss(Y_pred, Y_true)
        #ssim_loss_value = self.ssim_loss(Y_pred, Y_true)

        if torch.any(torch.isnan(saturation_loss)):
            print("NaN saturation loss!")
        if torch.any(torch.isnan(intensity_loss)):
            print("NaN intensity loss!")
        if torch.any(torch.isnan(spatial_variation_loss)):
            print("NaN spatial variation loss!")
        
        # loss= saturation_loss + intensity_loss + spatial_variation_loss + ssim_loss_value +dog_loss_value+log_loss_value 
        loss= saturation_loss + spatial_variation_loss + sobel_loss_value + intensity_loss + log_loss_value #+ ssim_loss_value

        # + sobel_loss_value
        return loss

def main(args):

    uciqe_values = []
    uqims_values=[]
    uqims_values=[]
    uicm_values=[]
    uism_values=[]
    uicomn_values=[]
    output_names = []

    seed = int(torch.randint(9223372036854775807, (1,))[0]) if args.seed is None else args.seed
    if args.seed is None:
        print('Seed:', seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    train_dataset = paired_rgb_depth_dataset(args.images, args.depth, args.depth_16u, args.mask_max_depth, args.height,
                                             args.width)
    save_dir = args.output
    # check_dir = args.checkpoints
    os.makedirs(save_dir, exist_ok=True)
    check_dir = args.checkpoints
    os.makedirs(check_dir, exist_ok=True)
    target_batch_size = args.batch_size
    dataloader = DataLoader(train_dataset, batch_size=target_batch_size, shuffle=False)
    bs_model = BackscatterNet().to(device)
    da_model = DeattenuateNet().to(device)
    bs_criterion = BackscatterLoss().to(device)
    da_criterion = DeattenuateLoss().to(device)
    bs_optimizer = torch.optim.Adam(bs_model.parameters(), lr=args.init_lr)
    da_optimizer = torch.optim.Adam(da_model.parameters(), lr=args.init_lr)
    skip_right = True
    total_bs_eval_time = 0.
    total_bs_evals = 0
    total_at_eval_time = 0.
    total_at_evals = 0
    for j, (left, depth, fnames) in enumerate(dataloader):
        print("training")
        image_batch = left
        batch_size = image_batch.shape[0]
        for iter in trange(args.init_iters if j == 0 else args.iters):  # Run first batch for 500 iters, rest for 50
            start = time()
            direct, backscatter = bs_model(image_batch, depth)
            bs_loss = bs_criterion(direct)
            bs_optimizer.zero_grad()
            bs_loss.backward()
            bs_optimizer.step()
            total_bs_eval_time += time() - start
            total_bs_evals += batch_size
        direct_mean = direct.mean(dim=[2, 3], keepdim=True)
        direct_std = direct.std(dim=[2, 3], keepdim=True)
        direct_z = (direct - direct_mean) / direct_std
        clamped_z = torch.clamp(direct_z, -5, 5)
        direct_no_grad = torch.clamp(
            (clamped_z * direct_std) + torch.maximum(direct_mean, torch.Tensor([1. / 255]).to(device)), 0, 1).detach()
        for iter in trange(args.init_iters if j == 0 else args.iters):  # Run first batch for 500 iters, rest for 50
            start = time()
            f, J = da_model(direct_no_grad, depth)
            da_loss = da_criterion(direct_no_grad, J)
            da_optimizer.zero_grad()
            da_loss.backward()
            da_optimizer.step()
            total_at_eval_time += time() - start
            total_at_evals += batch_size
        print("Losses: %.9f %.9f" % (bs_loss.item(), da_loss.item()))
        avg_bs_time = total_bs_eval_time / total_bs_evals * 1000
        avg_at_time = total_at_eval_time / total_at_evals * 1000
        avg_time = avg_bs_time + avg_at_time
        print("Avg time per eval: %f ms (%f ms bs, %f ms at)" % (avg_time, avg_bs_time, avg_at_time))
        img = image_batch.cpu()
        direct_img = torch.clamp(direct_no_grad, 0., 1.).cpu()
        backscatter_img = torch.clamp(backscatter, 0., 1.).detach().cpu()
        f_img = f.detach().cpu()
        f_img = f_img / f_img.max()
        J_img = torch.clamp(J, 0., 1.).cpu()
        for side in range(1 if skip_right else 2):
            side_name = 'left' if side == 0 else 'right'
            names = fnames[side]
            for n in range(batch_size):
                i = n + target_batch_size * side
                if args.save_intermediates:
                    save_image(direct_img[i], "%s/%s-direct.png" % (save_dir, names[n].rstrip('.png')))
                    save_image(backscatter_img[i], "%s/%s-backscatter.png" % (save_dir, names[n].rstrip('.png')))
                    save_image(f_img[i], "%s/%s-f.png" % (save_dir, names[n].rstrip('.png')))
                save_image(J_img[i], "%s/%s-corrected.png" % (save_dir, names[n].rstrip('.png')))
                output_image_path = "%s/%s-corrected.png" % (save_dir, names[n].rstrip('.png'))
                output_image =Image.open(output_image_path)
                output_image = output_image.resize((256, 256))
                image = output_image.convert('RGB')
                image_array = np.array(image)
                #output_image_np = np.array(output_image)
                #output_image_np = output_image_np.astype(np.float32)
                uciqe_value = getUCIQE(output_image_path)
                print('UCIQE:',uciqe_value)
                uqims_value,uicm_value,uism_value,uicomn_value =getUIQM(image_array)
                print('UQIMS:',uqims_value)
                uciqe_values.append(uciqe_value)
                uqims_values.append(uqims_value)
                uicm_values.append(uicm_value)
                uism_values.append(uism_value)
                uicomn_values.append(uicomn_value)
                output_names.append(names[n])

        # Save checkpoint with compression
        checkpoint_path = os.path.join(check_dir, f'model_checkpoint_{j}.pth')

        with gzip.open(checkpoint_path, 'wb') as f:
            torch.save({
            'bs_model_state_dict': bs_model.state_dict(),
            'da_model_state_dict': da_model.state_dict(),
            'bs_optimizer_state_dict': bs_optimizer.state_dict(),
            'da_optimizer_state_dict': da_optimizer.state_dict(),
            }, f)

        # torch.save({
        #     'bs_model_state_dict': bs_model.state_dict(),
        #     'da_model_state_dict': da_model.state_dict(),
        #     'bs_optimizer_state_dict': bs_optimizer.state_dict(),
        #     'da_optimizer_state_dict': da_optimizer.state_dict(),
        # }, os.path.join(check_dir, f'model_checkpoint_{j}.pth'))


            # Save to Excel
    df = pd.DataFrame({
        'Output Image Name': output_names,
        'uciqe': uciqe_values,
        'uqims': uqims_values,
        'uicm':uicm_values,
        'uism':uism_values,
        'uicomn':uicomn_values
        })
    
    excel_path = os.path.join(save_dir, 'evaluation_metrics.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Evaluation metrics saved to {excel_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='/home/user/Rajini/AUV_UW/Dataset/OceanLensExp/ExpDS/images_seethruWBGC14', help='Path to the images folder')
    parser.add_argument('--depth', type=str, default='/home/user/Rajini/AUV_UW/Dataset/OceanLensExp/ExpDS/depth_depthanyseathru' , help='Path to the depth folder')
    parser.add_argument('--output', type=str, default=f'/home/user/Rajini/AUV_UW/Output/2CNN_NBL_ABL_DA_output_OL_SEETHRU_2nn',  help='Path to the output folder')
    parser.add_argument('--checkpoints', type = str, default= '/home/user/Rajini/AUV_UW/Output/2CNN_NBL_ABL_DA_output_OL_SEETHRU_2nn/check')
    parser.add_argument('--height', type=int, default=1242, help='Height of the image and depth files')
    parser.add_argument('--width', type=int, default=1952, help='Width of the image and depth')
    parser.add_argument('--depth_16u', action='store_true',
                        help='True if depth images are 16-bit unsigned (millimetres), false if floating point (metres)')
    parser.add_argument('--mask_max_depth', action='store_true',
                        help='If true will replace zeroes in depth files with max depth')
    parser.add_argument('--seed', type=int, default=None, help='Seed to initialize network weights (use 1337 to replicate paper results)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing images')
    parser.add_argument('--save_intermediates', action='store_true', default=False, help='Set to True to save intermediate files (backscatter, attenuation, and direct images)')
    parser.add_argument('--init_iters', type=int, default=500, help='How many iterations to refine the first image batch (should be >= iters)')
    parser.add_argument('--iters', type=int, default=150, help='How many iterations to refine each image batch')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate for Adam optimizer')

    args = parser.parse_args()
    main(args)
