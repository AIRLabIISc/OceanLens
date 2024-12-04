import os
import torch
from scipy.ndimage import gaussian_laplace
from time import time
from torch.utils.data import Dataset
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


class paired_image_depth_data(Dataset): 
    def __init__(self, raw_image_path, depth_path, openni_depth, mask_max_depth, image_height, image_width):
        self.raw_image_folder = raw_image_path
        self.depth_folder = depth_path
        self.raw_image_files = sorted(os.listdir(raw_image_path))
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
        assert len(self.raw_image_files) == len(self.depth_files)

    def __len__(self):
        return len(self.raw_image_files)

    def __getitem__(self, index):
        frame_name = self.raw_image_files[index]
        raw_image = Image.open(os.path.join(self.raw_image_folder, frame_name))
        
        depth_frame_name = self.depth_files[index]
        depth = Image.open(os.path.join(self.depth_folder, depth_frame_name))
        
        # Check if the filenames match based on the first few characters
        if frame_name[:5] != depth_frame_name[:5]:
            raise ValueError(f"Mismatch between image and depth map: {frame_name} and {depth_frame_name}")
        
        # Print the image and corresponding depth map names
        print(f"Matched : Image: {frame_name}, Depth map: {depth_frame_name}")
        
        if depth.mode != 'L':
            depth_gray = depth.convert('L')
        else:
            depth_gray = depth
        
        depth_transformed: torch.Tensor = self.image_transforms(depth_gray).float().to(device)
        
        if self.openni_depth:
            depth_transformed = depth_transformed / 1000.
        if self.mask_max_depth:
            depth_transformed[depth_transformed == 0.] = depth_transformed.max()
        
        low, high = torch.nanquantile(depth_transformed, self.depth_perc), torch.nanquantile(depth_transformed, 1. - self.depth_perc)
        depth_transformed[(depth_transformed < low) | (depth_transformed > high)] = 0.
        
        depth_transformed = torch.squeeze(morph.closing(torch.unsqueeze(depth_transformed, dim=0), self.kernel), dim=0)
        
        image_transformed: torch.Tensor = self.image_transforms(raw_image).to(device) / 255.
        
        return image_transformed, depth_transformed, [frame_name]



