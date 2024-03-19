import torch
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn.functional as F
import torch.nn as nn

def psnr(img, imclean):
    # Ensure values are between 0 and 1
    img = torch.clamp(img, 0, 1)
    imclean = torch.clamp(imclean, 0, 1)

    # Convert to numpy arrays
    Img = img.cpu().numpy().astype(np.float32)
    Iclean = imclean.cpu().numpy().astype(np.float32)

    # Compute PSNR
    PSNR = []
    for i in range(Img.shape[0]):
        ps = compare_psnr(Iclean[i], Img[i], data_range=1.0)
        if not np.isinf(ps):
            PSNR.append(ps)
    return np.mean(PSNR)

def ssim(img, imclean):
    # Ensure values are between 0 and 1
    img = torch.clamp(img, 0, 1)
    imclean = torch.clamp(imclean, 0, 1)

    # Convert to numpy arrays
    Img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    Iclean = imclean.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)

    # 
    Iclean = np.squeeze(Iclean)
    Img = np.squeeze(Img)

    # Compute SSIM
    SSIM = np.mean([compare_ssim(Iclean[i], Img[i],data_range=1.0) for i in range(Img.shape[0])])

    return SSIM


mse = nn.MSELoss(reduction='mean') 

    
def loss_aug(clean, clean1, noise_w, noise_w1, noise_b, noise_b1):
    loss1 = mse(clean1,clean)
    loss2 = mse(noise_w1,noise_w)
    loss3 = mse(noise_b1,noise_b)
    loss = loss1 + loss2 + loss3
    return loss

def loss_main(input_noisy, input_noisy_pred, clean, clean1, clean2, clean3, noise_b, noise_b1, noise_b2, noise_b3, noise_w, noise_w1, noise_w2,std,gamma):
    loss1 = mse(input_noisy_pred, input_noisy)
    
    loss2 = mse(clean1,clean)
    loss3 = mse(noise_b3, noise_b)
    loss4 = mse(noise_w2, noise_w)
    loss5 = mse(clean2, clean)
    
    loss6 = mse(clean3, torch.zeros_like(clean3))
    loss7 = mse(noise_w1, torch.zeros_like(noise_w1))
    loss8 = mse(noise_b1, torch.zeros_like(noise_b1))
    loss9 = mse(noise_b2, torch.zeros_like(noise_b2))

    sigma_b = torch.std(noise_b.reshape([noise_b.shape[0],noise_b.shape[1],-1]),-1)
    sigma_w = torch.std(noise_w.reshape([noise_w.shape[0],noise_w.shape[1],-1]),-1)
    blur_clean = F.avg_pool2d(clean, kernel_size=6, stride=1, padding=3) 
    clean_mean = torch.mean(torch.square(torch.pow(blur_clean,gamma).reshape([clean.shape[0],clean.shape[1],-1])),-1)#.detach()
    sigma_wb = torch.sqrt(clean_mean*torch.square(sigma_w)+torch.square(sigma_b))
    
    loss10 = mse(sigma_wb,std)

    loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10
    return loss

