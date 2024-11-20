# OceanLens 

Underwater environments present significant challenges due to the selective absorption and scattering of light by water, which affects image clarity, contrast, and color fidelity. In this paper, we tackle these challenges with OceanLens. We model underwater image physics, including backscatter and attenuation, using neural networks with new loss functions such as adaptive backscatter loss and edge correction losses that address variance and luminance. We also demonstrate the relevance of pre-trained monocular depth estimation models for generating underwater depth maps. Our evaluation compares the performance of various loss functions against state-of-the-art methods using the "SeeThru" dataset, revealing significant improvements. Specifically, we observe a roughly 65% reduction in Grayscale Patch Mean Angular Error (GPMAE) and a 60% increase in the Underwater Image Quality Metric (UIQM) compared to the SeeThru and DeepSeeColor methods. Two additional convolutional layers are added to capture subtle image details more effectively with OceanLens. This architecture is validated on the UIEB dataset, with model performance assessed using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) metrics. OceanLens with 2-layer CNN achieves up to a 12-15% improvement in SSIM and a marginal improvement in PSNR.   
To demonstrate the efficacy of OceanLens, we evaluated underwater image enhancement using three datasets: See-Thru, US Virgin Islands, and UIEB. These datasets, each with unique underwater conditions and image quality challenges, are ideal for testing our techniques. For depth maps, we utilized original maps and those generated by Monodepth2 and Depth-Anything-V2-Large, originally trained on terrestrial data. Our pre-processing involved white balancing and gamma correction. All models are implemented using PyTorch and are trained on an NVIDIA GeForce RTX 4090 GPU. On average, OceanLens takes approximately 4-5 milliseconds to process images with sizes ranging between 7 to 12 MB.  

Watch the Demonstration of our work on the above-mentioned datasets given in the  link below: [Click here to Dive into the video !](https://drive.google.com/drive/folders/1ekX5J3ZiYKjqTK49yKgqYoCv-ASneabM?usp=sharing)  

To access and download the high-resolution images used in this project, visit the following link: [Explore High-Resolution Images here !](https://drive.google.com/drive/folders/1qQ6tYtNdvts17eMQOew_XH8ubdNWu5ZG)