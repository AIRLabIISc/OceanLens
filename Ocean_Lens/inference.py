import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import gzip
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np

# Import the dataset and model classes from the training script
from train import paired_image_depth_data
from train import BackscatterNet, DeattenuateNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(checkpoint_path, bs_model, da_model, bs_optimizer, da_optimizer):
    with gzip.open(checkpoint_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    bs_model.load_state_dict(checkpoint['bs_model_state_dict'])
    da_model.load_state_dict(checkpoint['da_model_state_dict'])
    bs_optimizer.load_state_dict(checkpoint['bs_optimizer_state_dict'])
    da_optimizer.load_state_dict(checkpoint['da_optimizer_state_dict'])
    return bs_model, da_model, bs_optimizer, da_optimizer

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def gray_world(image, balance_factor=1.0):
    global_mean = image.mean(axis=(0, 1))
    image_grayworld = (image * (global_mean.mean() / global_mean) ** balance_factor).clip(0, 255).astype(np.uint8)
    if image.shape[2] == 4:
        image_grayworld[:, :, 3] = 255
    return image_grayworld

def preprocess_images(images_path, output_path, gamma_value, wb_factor, preprocess_option):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(images_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                if preprocess_option == "Gamma Correction" or preprocess_option == "Both":
                    image = gamma_correction(image, gamma_value)
                if preprocess_option == "White Balancing" or preprocess_option == "Both":
                    image = gray_world(image, wb_factor)

                output_file = os.path.join(output_path, filename)
                cv2.imwrite(output_file, image)
                print(f'Processed and saved: {output_file}')
            else:
                print(f'Failed to load: {image_path}')
        else:
            print(f'Skipped non-image file: {filename}')

def run_inference(images_path, depth_path, checkpoints_path, height, width, init_lr, preprocess_option, gamma_value, wb_factor):
    try:
        output_path = os.path.join(os.path.dirname(images_path), "inference_output")
        os.makedirs(output_path, exist_ok=True)

        # Preprocess images if required
        temp_output_path = None  # Initialize temp_output_path for later use
        if preprocess_option != "None":
            temp_output_path = os.path.join(os.path.dirname(images_path), "preprocessed_images")
            preprocess_images(images_path, temp_output_path, gamma_value, wb_factor, preprocess_option)
            images_path = temp_output_path  # Use preprocessed images for inference

        bs_model = BackscatterNet().to(device)
        da_model = DeattenuateNet().to(device)
        bs_optimizer = torch.optim.Adam(bs_model.parameters(), lr=init_lr)
        da_optimizer = torch.optim.Adam(da_model.parameters(), lr=init_lr)

        checkpoint_path = sorted(os.listdir(checkpoints_path))[-1]
        checkpoint_path = os.path.join(checkpoints_path, checkpoint_path)
        bs_model, da_model, bs_optimizer, da_optimizer = load_checkpoint(checkpoint_path, bs_model, da_model, bs_optimizer, da_optimizer)

        inference_dataset = paired_image_depth_data(images_path, depth_path, False, False, height, width)
        dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

        bs_model.eval()
        da_model.eval()

        with torch.no_grad():
            for i, (left, depth, framenames) in enumerate(dataloader):
                image_batch = left.to(device)
                depth = depth.to(device)
                I_D, backscatter_comp = bs_model(image_batch, depth)
                correction, I = da_model(I_D, depth)

                I_D_img = torch.clamp(I_D, 0., 1.).cpu()
                backscatter_img = torch.clamp(backscatter_comp, 0., 1.).cpu()
                correction_img = correction.detach().cpu()
                correction_img = correction_img / correction_img.max()
                I_img = torch.clamp(I, 0., 1.).cpu()

                for n in range(image_batch.size(0)):
                    framename = framenames[0][n]
                    save_image(I_D_img[n], os.path.join(output_path, f'{framename.rstrip(".png")}-direct.png'))
                    save_image(backscatter_img[n], os.path.join(output_path, f'{framename.rstrip(".png")}-backscatter.png'))
                    save_image(correction_img[n], os.path.join(output_path, f'{framename.rstrip(".png")}-attenuation.png'))
                    save_image(I_img[n], os.path.join(output_path, f'{framename.rstrip(".png")}-corrected.png'))

        messagebox.showinfo("Success", f"Inference completed successfully! Results saved in {output_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def select_folder(entry):
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry.delete(0, tk.END)
        entry.insert(0, folder_path)

def toggle_factor_entries(*args):
    option = preprocess_option.get()
    # Enable or disable the gamma entry based on the selection
    if option == "Gamma Correction":
        gamma_entry.config(state=tk.NORMAL)
        wb_entry.config(state=tk.DISABLED)
    elif option == "White Balancing":
        gamma_entry.config(state=tk.DISABLED)
        wb_entry.config(state=tk.NORMAL)
    elif option == "Both":
        gamma_entry.config(state=tk.NORMAL)
        wb_entry.config(state=tk.NORMAL)
    else:  # None
        gamma_entry.config(state=tk.DISABLED)
        wb_entry.config(state=tk.DISABLED)

def create_gui():
    global preprocess_option, gamma_entry, wb_entry

    root = tk.Tk()
    root.title("OceanLens")  # Creative title
    root.geometry("700x600")  # Increased window size
    root.configure(bg="white")
    root.attributes('-alpha', 0.9)  # Make background slightly transparent

    # Style configuration
    style = ttk.Style(root)
    style.configure("TLabel", font=("Cambria", 12, "italic"), background="white")
    style.configure("TButton", font=("Cambria", 12, "italic"), padding=5)
    style.configure("TEntry", font=("Cambria", 12), padding=5)
    style.configure("TFrame", background="lightblue")  # Set frame background color
    style.configure("TNotebook", background="lightblue")  # Set tab background color
    style.configure("TNotebook.Tab", background="lightblue", padding=[10, 5])

    title_label = ttk.Label(root, text="OceanLens: Enhancing Underwater Vision", font=("Cambria", 16, "italic"))  # Tagline
    title_label.pack(pady=10)

    frame = ttk.Frame(root, padding=20)
    frame.pack(expand=True, fill=tk.BOTH)

    ttk.Label(frame, text="Images Folder:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    images_entry = ttk.Entry(frame, width=50)
    images_entry.grid(row=0, column=1, padx=5, pady=5)
    ttk.Button(frame, text="Browse", command=lambda: select_folder(images_entry)).grid(row=0, column=2, padx=5, pady=5)

    ttk.Label(frame, text="Depth Folder:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    depth_entry = ttk.Entry(frame, width=50)
    depth_entry.grid(row=1, column=1, padx=5, pady=5)
    ttk.Button(frame, text="Browse", command=lambda: select_folder(depth_entry)).grid(row=1, column=2, padx=5, pady=5)

    ttk.Label(frame, text="Checkpoints Folder:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    checkpoints_entry = ttk.Entry(frame, width=50)
    checkpoints_entry.grid(row=2, column=1, padx=5, pady=5)
    ttk.Button(frame, text="Browse", command=lambda: select_folder(checkpoints_entry)).grid(row=2, column=2, padx=5, pady=5)

    ttk.Label(frame, text="Height:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    height_entry = ttk.Entry(frame)
    height_entry.insert(0, "1242")
    height_entry.grid(row=3, column=1, padx=5, pady=5)

    ttk.Label(frame, text="Width:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
    width_entry = ttk.Entry(frame)
    width_entry.insert(0, "1952")
    width_entry.grid(row=4, column=1, padx=5, pady=5)

    ttk.Label(frame, text="Initial Learning Rate:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
    lr_entry = ttk.Entry(frame)
    lr_entry.insert(0, "0.01")
    lr_entry.grid(row=5, column=1, padx=5, pady=5)

    # Preprocessing options
    ttk.Label(frame, text="Preprocessing:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
    preprocess_option = tk.StringVar(value="")
    ttk.Radiobutton(frame, text="None", variable=preprocess_option, value="None").grid(row=6, column=1, sticky=tk.W)
    ttk.Radiobutton(frame, text="White Balancing", variable=preprocess_option, value="White Balancing").grid(row=7, column=1, sticky=tk.W)
    ttk.Radiobutton(frame, text="Gamma Correction", variable=preprocess_option, value="Gamma Correction").grid(row=8, column=1, sticky=tk.W)
    ttk.Radiobutton(frame, text="Both", variable=preprocess_option, value="Both").grid(row=9, column=1, sticky=tk.W)

    # Labels and Entries for Gamma and WB factor
    ttk.Label(frame, text="Gamma Value:").grid(row=10, column=0, sticky=tk.W, padx=5, pady=5)
    gamma_entry = ttk.Entry(frame)
    gamma_entry.insert(0, "1.5")  # Default gamma value
    gamma_entry.grid(row=10, column=1, padx=5, pady=5)

    ttk.Label(frame, text="WB Factor:").grid(row=11, column=0, sticky=tk.W, padx=5, pady=5)
    wb_entry = ttk.Entry(frame)
    wb_entry.insert(0, "1.0")  # Default WB factor
    wb_entry.grid(row=11, column=1, padx=5, pady=5)
    # Trace the preprocess_option variable to toggle factor entries
    preprocess_option.trace("w", toggle_factor_entries)

    def start_inference():
        images_path = images_entry.get()
        depth_path = depth_entry.get()
        checkpoints_path = checkpoints_entry.get()
        height = int(height_entry.get())
        width = int(width_entry.get())
        init_lr = float(lr_entry.get())
        gamma_value = float(gamma_entry.get())
        wb_factor = float(wb_entry.get())

        run_inference(images_path, depth_path, checkpoints_path, height, width, init_lr, preprocess_option.get(), gamma_value, wb_factor)

    ttk.Button(frame, text="Start Inference", command=start_inference).grid(row=12, column=1, padx=5, pady=20)

    root.mainloop()

if __name__ == '__main__':
    create_gui()
