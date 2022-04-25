import sys
sys.path.append('/projects/img/fgx/ARLigand/GANData/')
import numpy as np
from numpy import save

SSIM_train, DICE_train, MAE_train, MSE_train, PSNR_train, LOSS_train = {}, {}, {}, {}, {}, {}
SSIM_val, DICE_val, MAE_val, MSE_val, PSNR_val, LOSS_val = {}, {}, {}, {}, {}, {}

for channel in range(5):

    SSIM_train[channel] = np.array([])
    DICE_train[channel] = np.array([])
    MAE_train[channel] = np.array([])
    MSE_train[channel] = np.array([])
    PSNR_train[channel] = np.array([])
    
    SSIM_val[channel] = np.array([])
    DICE_val[channel] = np.array([])
    MAE_val[channel] = np.array([])
    MSE_val[channel]= np.array([])
    PSNR_val[channel] = np.array([])
    
    save(f'SSIM_train_{channel}.npy', SSIM_train[channel])
    save(f'DICE_train_{channel}.npy', DICE_train[channel])
    save(f'MAE_train_{channel}.npy', MAE_train[channel])
    save(f'MSE_train_{channel}.npy', MSE_train[channel])
    save(f'PSNR_train_{channel}.npy', PSNR_train[channel])
    
    save(f'SSIM_val_{channel}.npy', SSIM_val[channel])
    save(f'DICE_val_{channel}.npy', DICE_val[channel])
    save(f'MAE_val_{channel}.npy', MAE_val[channel])
    save(f'MSE_val_{channel}.npy', MSE_val[channel])
    save(f'PSNR_val_{channel}.npy', PSNR_val[channel])
    
LOSS_train = np.array([])
LOSS_val = np.array([])
save(f'LOSS_train.npy', LOSS_train)
save(f'LOSS_val.npy', LOSS_val)



