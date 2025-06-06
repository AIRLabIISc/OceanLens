import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import kornia.morphology as morph

import torch
import gzip


from Final_Programs.OceanLens import paired_rgb_depth_dataset, BackscatterNet, DeattenuateNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
def load_checkpoint(checkpoint_path, bs_model, da_model, bs_optimizer, da_optimizer):
    # Open the checkpoint file with gzip for decompression
    with gzip.open(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f)
    
    # Load the state dictionaries into the models and optimizers
    bs_model.load_state_dict(checkpoint['bs_model_state_dict'])
    da_model.load_state_dict(checkpoint['da_model_state_dict'])
    bs_optimizer.load_state_dict(checkpoint['bs_optimizer_state_dict'])
    da_optimizer.load_state_dict(checkpoint['da_optimizer_state_dict'])
    
    return bs_model, da_model, bs_optimizer, da_optimizer
# def load_checkpoint(checkpoint_path, bs_model, da_model, bs_optimizer, da_optimizer):
#     # Load directly without gzip
#     checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)

#     bs_model.load_state_dict(checkpoint['bs_model_state_dict'])
#     da_model.load_state_dict(checkpoint['da_model_state_dict'])
#     bs_optimizer.load_state_dict(checkpoint['bs_optimizer_state_dict'])
#     da_optimizer.load_state_dict(checkpoint['da_optimizer_state_dict'])
    
#     return bs_model, da_model, bs_optimizer, da_optimizer

def main(args):
    # Load models
    bs_model = BackscatterNet().to(device)
    da_model = DeattenuateNet().to(device)
    bs_optimizer = torch.optim.Adam(bs_model.parameters(), lr=args.init_lr)
    da_optimizer = torch.optim.Adam(da_model.parameters(), lr=args.init_lr)
    save_dir = args.output
    
    os.makedirs(save_dir, exist_ok=True)
    # Load the latest checkpoint
    checkpoint_path = sorted(os.listdir(args.checkpoints))[-1]
    checkpoint_path = os.path.join(args.checkpoints, checkpoint_path)
    bs_model, da_model, bs_optimizer, da_optimizer = load_checkpoint(checkpoint_path, bs_model, da_model, bs_optimizer, da_optimizer)

    # Prepare the dataset and dataloader for inference
    inference_dataset = paired_rgb_depth_dataset(args.images, args.depth, args.depth_16u, args.mask_max_depth, args.height, args.width)
    dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
    
    bs_model.eval()
    da_model.eval()
    
    with torch.no_grad():
        for i, (left, depth, fnames) in enumerate(dataloader):
            image_batch = left.to(device)
            depth = depth.to(device)

            # Perform inference
            direct, backscatter = bs_model(image_batch, depth)
            f, J = da_model(direct, depth)
            
            # Process and save results
            direct_img = torch.clamp(direct, 0., 1.).cpu()
            backscatter_img = torch.clamp(backscatter, 0., 1.).cpu()
            f_img = f.detach().cpu()
            f_img = f_img / f_img.max()
            J_img = torch.clamp(J, 0., 1.).cpu()
            
            for n in range(image_batch.size(0)):
                fname = fnames[0][n]
                save_image(direct_img[n], os.path.join(args.output, f'{fname.rstrip(".png")}-direct.png'))
                save_image(backscatter_img[n], os.path.join(args.output, f'{fname.rstrip(".png")}-backscatter.png'))
                save_image(f_img[n], os.path.join(args.output, f'{fname.rstrip(".png")}-f.png'))
                save_image(J_img[n], os.path.join(args.output, f'{fname.rstrip(".png")}-corrected.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--images', type=str, default='', help='Path to the images folder')
    parser.add_argument('--depth', type=str, default='/home/user/Rajini/AUV_UW/Dataset/OceanLensExp/ExpDS/depth_depthanyseathru' , help='Path to the depth folder')
    
    parser.add_argument('--output', type=str, default=f'/home/user/Rajini/AUV_UW/Output/2CNN_NBL_ABL_DA_output_all_Seethru_check6',  help='Path to the output folder')
    parser.add_argument('--checkpoints', type = str, default= '/home/user/Rajini/AUV_UW/Output/2CNN_NBL_ABL_DA_output_OL_SEETHRU_2nn/check1')
    parser.add_argument('--height', type=int, default=720, help='Height of the image and depth files')
    #parser.add_argument('--height', type=int, default=1242, help='Height of the image and depth files')
    parser.add_argument('--width', type=int, default=1280, help='Width of the image and depth files')
    parser.add_argument('--depth_16u', action='store_true', help='True if depth images are 16-bit unsigned (millimetres), false if floating point (metres)')
    parser.add_argument('--mask_max_depth', action='store_true', help='If true will replace zeroes in depth files with max depth')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate for Adam optimizer')

    args = parser.parse_args()
    main(args)
