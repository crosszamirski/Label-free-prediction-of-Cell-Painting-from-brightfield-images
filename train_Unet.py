import torch
import torch.optim 
from torch.utils.data import DataLoader
import h5py
from Networks.unet2d import UNet
from Utils.UnetLoss import GenLoss
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy import save
from torch.utils.tensorboard import SummaryWriter
import time
import math
import tables
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys
sys.path.append('.../{your_directory}/')

dataname="cell" 
ignore_index = 0 # his value won't be included in the loss calculation (output image value)- e.g. 0 is good for this data.
gpuid=0

#Unet params
n_classes= 5    #output channels (fluorescent)
in_channels= 3  #input channels (brightfield)
padding= True   #should levels be padded
depth= 6     #depth of the network 
wf= 5           #wf (int): number of filters in the first layer is 2**wf, was 6
up_mode= 'upconv' #upsample or interpolation 
batch_norm = False #sbatch normalization between the layers

#Training params
batch_size=20
patch_size=256
num_epochs = 500
edge_weight = 1.1 
phases = ["train","val"] 
validation_phases= ["val"] 

#specify if we should use a GPU (cuda) or only the CPU
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')

Gen = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
#print(f"total params: \t{sum([np.prod(p.size()) for p in Gen.parameters()])}")

class Dataset(object):
    def __init__(self, fname ,img_transform=None, mask_transform = None, edge_weight= False):
        self.fname=fname
        self.edge_weight = edge_weight
        self.img_transform=img_transform
        self.mask_transform = mask_transform
        self.tables=tables.open_file(self.fname)
        self.numpixels=self.tables.root.numpixels[:]
        self.nitems=self.tables.root.img.shape[0]
        self.tables.close()
        
        self.img = None
        self.mask = None
        
    def __getitem__(self, index):
        with tables.open_file(self.fname,'r') as db:
            self.img=db.root.img
            self.mask=db.root.mask
            mask = self.mask[index,:,:,:]
            img = self.img[index,:,:,:]
            
        img_new = img
        return img_new, mask#, weight_new
    def __len__(self):
        return self.nitems

tables.file._open_files.close_all()

dataset, dataset2={}
dataLoader={}
dataLoader2={}
for phase in phases: #now for each of the phases, we're creating the dataloader
                     #interestingly, given the batch size, i've not seen any improvements from using a num_workers>0
    f = h5py.File(f"./{dataname}_{phase}.pytable")
    f.close()
    dataset[phase]=Dataset(f"./{dataname}_{phase}.pytable")
    dataLoader[phase]=DataLoader(dataset[phase], batch_size=batch_size, 
                                shuffle=True, num_workers=0, pin_memory=False)
    tables.file._open_files.close_all()

optimizerG = torch.optim.Adam(Gen.parameters(),lr=.0002)
optim = torch.optim.Adam(Gen.parameters(), 
                           lr=.0002,
                           weight_decay=0.0002)

nclasses = dataset["train"].numpixels.shape[1]
gen_criterion = GenLoss()

writer=SummaryWriter() 
best_loss_on_test = np.Infinity
edge_weight=torch.tensor(edge_weight).to(device)
start_time = time.time()

checkpoint = torch.load(f"{dataname}_epoch_30_Unet.pth")
start_epoch = checkpoint['epoch']
Gen.load_state_dict(checkpoint['model_dict'])

#Save some variables e.g. MAE, SSIM etc
#The blank arrays are defined in file called 'make_variable_table.py'
SSIM_train, DICE_train, MAE_train, MSE_train, PSNR_train, LOSS_train = {}, {}, {}, {}, {}
SSIM_val, DICE_val, MAE_val, MSE_val, PSNR_val, LOSS_val = {}, {}, {}, {}, {}

def PSNR(im1, im2):
    im1 = im1.astype(np.float64) / 255
    im2 = im2.astype(np.float) / 255
    mse = np.mean((im1 - im2)**2)
    return 10*math.log10(1. / mse)

loss_values, loss_values_val = [], []
running_loss, running_loss_val = 0.0, 0.0
for epoch in range(start_epoch, num_epochs):
    all_acc = {key: 0 for key in phases} 
    all_loss = {key: torch.zeros(0).to(device) for key in phases}
    cmatrix = {key: np.zeros((2,2)) for key in phases}

    for ii , (X, y) in enumerate(dataLoader["train"]): #for each of the batches
        Gen.train()
        y = y.to(device)
        X = X.to(device)                 
        prediction = Gen(X)
        Gen.zero_grad()
        g_loss = gen_criterion(prediction, y, epoch) 
        g_loss.backward(retain_graph=True)
        prediction = Gen(X)
        optimizerG.step()
        
        if ii % 10 == 0:
            # save metrics to their arrays every 10 updates
            # save model every 10 updates
           
            state = {'epoch' : epoch +1,
            'model_dict': Gen.state_dict(),
            'optim_dict': optim.state_dict(),
            'best_loss_on_test': all_loss,
            'n_classes': n_classes,
            'in_channels': in_channels,
            'padding': padding,
            'depth': depth,
            'wf': wf,
            'up_mode': up_mode, 'batch_norm': batch_norm}
        
            torch.save(state, f"{dataname}_epoch_{epoch}_Unet.pth")
            
#            loss_train = g_loss.cpu().detach().numpy()#np.asarray(g_np)
#            LOSS_train = np.append(LOSS_train, loss_train)
#            save(f'LOSS_train.npy', LOSS_train)
#            
            for channel in range(5):
                               
                hold_pred = prediction[0,channel,:,:]
                prediction = hold_pred.detach().numpy()
                hold_gt = y[0,channel,:,:]
                ground_truth = hold_gt.detach().numpy()
                ssim_train = ssim(prediction, ground_truth)
                mse_train = mean_squared_error(prediction, ground_truth)
                mae_train = mean_absolute_error(prediction, ground_truth)
                psnr_train = PSNR(prediction, ground_truth)
                print('TRAINING: SSIM: ',ssim_train, ' MSE: ',mse_train, ' MAE: ',mae_train, ' PSNR: ', psnr_train)
#                SSIM_train[channel] = np.append(SSIM_train[channel], ssim_train)
#                save(f'SSIM_train_{channel}.npy', SSIM_train[channel])
#                MSE_train[channel] = np.append(MSE_train[channel], mse_train)
#                save(f'MSE_train_{channel}.npy', MSE_train[channel])
#                MAE_train[channel] = np.append(MAE_train[channel], mae_train)
#                save(f'MAE_train_{channel}.npy', MAE_train[channel])
#                PSNR_train[channel] = np.append(PSNR_train[channel], psnr_train)
#                save(f'PSNR_train_{channel}.npy', PSNR_train[channel])

            
    for jj , (X, y) in enumerate(dataLoader["val"]):
        Gen.eval()
        y = y.to(device)
        X = X.to(device)
        prediction1 = Gen(X)
        for channel in range(5):
            hold_pred = prediction1[0,channel,:,:]
            prediction = hold_pred.detach().numpy() # prediction 
            hold_gt = y[0,channel,:,:]
            ground_truth = hold_gt.detach().numpy() # GT

            ssim_val = ssim(prediction, ground_truth)
            mse_val = mean_squared_error(prediction, ground_truth)
            mae_val = mean_absolute_error(prediction, ground_truth)
            psnr_val = PSNR(prediction, ground_truth)   
            print('VALIDATION: SSIM: ',ssim_val, ' MSE: ',mse_val, ' MAE: ',mae_val, ' PSNR: ', psnr_val)
#            SSIM_val[channel] = np.append(SSIM_val[channel], ssim_val)
#            save(f'SSIM_val_{channel}.npy', SSIM_val[channel])
#            MSE_val[channel] = np.append(MSE_val[channel], mse_val)
#            save(f'MSE_val_{channel}.npy', MSE_val[channel])
#            MAE_val[channel] = np.append(MAE_val[channel], mae_val)
#            save(f'MAE_val_{channel}.npy', MAE_val[channel])
#            PSNR_val[channel] = np.append(PSNR_val[channel], psnr_val)
#            save(f'PSNR_val_{channel}.npy', PSNR_val[channel])
