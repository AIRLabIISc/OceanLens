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

from adaptive_backscatter_network import BackscatterNet
from deattenuation_network import DeattenuateNet
from adaptive_backscatter_loss import BackscatterLoss
from deattenuation_loss import DeattenuateLoss
from data_loader import paired_image_depth_data
from evaluation_metrices import getUCIQE, getUIQM
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd



# Device definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    uciqe_values = []
    uqims_values=[]
    uqims_values=[]
    uicm_values=[]
    uism_values=[]
    uicomn_values=[]
    output_names = []

    print('Training 1st batch for 500 epochs..') 
    train_dataset = paired_image_depth_data(args.images, args.depth, args.depth_16u, args.mask_max_depth, args.height,
                                             args.width)
    save_dir = args.output
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
    for j, (left, depth, frame_names) in enumerate(dataloader):
        print("training")
        image_batch = left
        batch_size = image_batch.shape[0]
        for iter in trange(args.init_iters if j == 0 else args.iters):  # Run first batch for 500 iters, rest for 50
            start = time()
            I_D, backscatter = bs_model(image_batch, depth)
            bs_loss = bs_criterion(I_D)
            bs_optimizer.zero_grad()
            bs_loss.backward()
            bs_optimizer.step()
            total_bs_eval_time += time() - start
            total_bs_evals += batch_size
        I_D_mean = I_D.mean(dim=[2, 3], keepdim=True)
        I_D_std = I_D.std(dim=[2, 3], keepdim=True)
        I_D_z = (I_D - I_D_mean) / I_D_std
        clamped_z = torch.clamp(I_D_z, -5, 5)
        I_D_no_grad = torch.clamp(
            (clamped_z * I_D_std) + torch.maximum(I_D_mean, torch.Tensor([1. / 255]).to(device)), 0, 1).detach()
        for iter in trange(args.init_iters if j == 0 else args.iters):  # Run first batch for 500 iters, rest for 50
            start = time()
            correction_factor_depth, I = da_model(I_D_no_grad, depth)
            da_loss = da_criterion(I_D_no_grad, I)
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
        I_D_img = torch.clamp(I_D_no_grad, 0., 1.).cpu()
        backscatter_img = torch.clamp(backscatter, 0., 1.).detach().cpu()
        correction_factor_depth_img = correction_factor_depth.detach().cpu()
        correction_factor_depth_img = correction_factor_depth_img / correction_factor_depth_img.max()
        I_img = torch.clamp(I, 0., 1.).cpu()
        for side in range(1 if skip_right else 2):
            side_name = 'left' if side == 0 else 'right'
            names = frame_names[side]
            for n in range(batch_size):
                i = n + target_batch_size * side
                if args.save_intermediates:
                    save_image(I_D_img[i], "%s/%s-direct.png" % (save_dir, names[n].rstrip('.png')))
                    save_image(backscatter_img[i], "%s/%s-backscatter.png" % (save_dir, names[n].rstrip('.png')))
                    save_image(correction_factor_depth_img[i], "%s/%s-attenuation.png" % (save_dir, names[n].rstrip('.png')))
                save_image(I_img[i], "%s/%s-corrected.png" % (save_dir, names[n].rstrip('.png')))
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

    parser.add_argument('--images', type=str, default='D:\\Users\\dtsmv\\Downloads\\pair\\Original\\Original', help='Path to the images folder')
    parser.add_argument('--depth', type=str, default='D:\\Users\\dtsmv\\Downloads\\pair\\DepthAny\\DepthAny' , help='Path to the depth folder')
    parser.add_argument('--output', type=str, default=f'D:\\Users\\dtsmv\\Downloads\\pair\\out_log', help='Path to the output folder')
    parser.add_argument('--checkpoints', type = str, default= 'D:\\Users\\dtsmv\\Downloads\\pair\\out_log\\check', help='Path to the checkpoints folder') 
    parser.add_argument('--height', type=int, default=1242, help='Height of the image and depth files')
    parser.add_argument('--width', type=int, default=1952, help='Width of the image and depth')
    parser.add_argument('--depth_16u', action='store_true',
                        help='True if depth images are 16-bit unsigned (millimetres), false if floating point (metres)')
    parser.add_argument('--mask_max_depth', action='store_true',
                        help='If true will replace zeroes in depth files with max depth')
    parser.add_argument('--seed', type=int, default=None, help='Seed to initialize network weights (use 1337 to replicate paper results)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing images')
    parser.add_argument('--save_intermediates', action='store_true', default=False, help='Set to True to save intermediate files (backscatter, attenuation, and direct images)')
    parser.add_argument('--init_iters', type=int, default=50, help='How many iterations to refine the first image batch (should be >= iters)')
    parser.add_argument('--iters', type=int, default=500, help='How many iterations to refine each image batch')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate for Adam optimizer')

    args = parser.parse_args()
    main(args)
