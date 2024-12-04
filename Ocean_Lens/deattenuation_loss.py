import torch
import torch.nn as nn
import cv2 
import numpy as np



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

    
    def forward(self, I_D, I):
        L_saturation = (self.relu(-I) + self.relu(I - 1)).square().mean()
        init_spatial = torch.std(I_D, dim=[2, 3])
        channel_intensities = torch.mean(I, dim=[2, 3], keepdim=True)
        channel_spatial = torch.std(I, dim=[2, 3])
        L_intensity = (channel_intensities - self.target_intensity).square().mean()
        L_spatial_variation = self.mse(channel_spatial, init_spatial)
        Y_pred = I.cpu().detach()
        Y_true = I_D.cpu().detach()
        
        L_sobel = self.sobel_edge_loss(Y_pred, Y_true)
        L_log = self.log_loss(Y_pred, Y_true)
        

        if torch.any(torch.isnan(L_saturation)):
            print("NaN saturation loss!")
        if torch.any(torch.isnan(L_intensity)):
            print("NaN intensity loss!")
        if torch.any(torch.isnan(L_spatial_variation)):
            print("NaN spatial variation loss!")
        
        loss= L_saturation + L_spatial_variation + L_sobel + L_intensity + L_log 

    
        return loss
