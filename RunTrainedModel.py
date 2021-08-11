from unet2d import UNet
import torch
import torch.optim 
import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
from PIL import Image
import sys, glob
sys.path.append('.../{your_directory}/')

dataname="cell"
# Load the saved model weights e.g. 28th epoch of trained WGAN model
checkpoint = torch.load(f"{dataname}_28Wgan.pth") 

# Network/Training Parameters (copied from training)
ignore_index = 0 
gpuid=0
n_classes= 5
in_channels= 3
padding= True
depth= 6
wf= 5 
up_mode= 'upconv' 
batch_norm = False 
batch_size=1
patch_size=256
edge_weight = 1.1 
phases = ["train","val"] 
validation_phases= ["val"] 

# Specify if we should use a GPU (cuda) or only the CPU
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    device = torch.device(f'cpu')

# Define the network
Gen = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,depth=depth,wf=wf, up_mode=up_mode, batch_norm=batch_norm).to(device)
print(f"total params: \t{sum([np.prod(p.size()) for p in Gen.parameters()])}")
Gen.load_state_dict(checkpoint['model_dict'])
Gen.eval()

# Define empty arrays
checkfull = {}
a = {}
b2 = {}
gt2 = {}
gt3 = {}
ionorm = {}
blank_bf = np.zeros((3,1024, 1024), dtype=np.float32)          
blank_fl = np.zeros((5, 1024, 1024))#, dtype=np.uint32)

# Load the test set
files_fluor = sorted(glob.glob (".../fluor/*.npy"))
files_bf = sorted(glob.glob (".../brightfield/*.npy")) 
    

# Loop through each file in the test set jj
for jj in range(len(files_fluor)):        
    namehold = files_fluor[jj]
    name_fluor = namehold[-20:-4] # we select the character string to suit the naming convention for this data
    
    # Load the ground truth fluorescent image (unused, just to save as .tif)
    gtim1 = np.load(files_fluor[jj])

    # Load and resize the brightfield input
    X1load = np.load(files_bf[jj]) 
    X1 = cv2.resize(X1load, dsize=(998,998), interpolation=cv2.INTER_CUBIC)
    X1 = np.swapaxes(X1,0,2)
    X1 = np.swapaxes(X1,1,2)
    X1 = np.expand_dims(X1, axis = 0)

    # Normalise the brightfield channels (each channel to have mean of zero and s.d. of one)
    for channel in range(3):
        X1a = X1[:,channel,:,:]
        mean, std = X1a.mean(), X1a.std()
        b2[channel] = (X1a-mean)/std
        b2[channel] = np.expand_dims(b2[channel],axis = 0)
        b29 = b2[channel]
        gtim69 = Image.fromarray(b29[0,0,:,:])
        if channel>0:
            ionorm = np.concatenate((ionorm,b2[channel]),axis = 0)
        else:
            ionorm = b2[0]
    X1 = np.swapaxes(ionorm,0,1)
    blank_bf[:,0:998,0:998] = X1[0,:,:,:]
    X = blank_bf # X is the 3 channel 998x998 normalised brightfield input
    
    
    # These two loops execute the stitching algorithm (each pixel is the median of four overlapping 256x256 tiles)
    countP = 0
    for x in range(7):
        for y in range(7):
            x_in = X[:,x*128:(256+x*128), y*128:(256+y*128)]
            x_in = np.expand_dims(x_in,axis = 0)
            x_in = torch.from_numpy(x_in)
            prediction1 = Gen(x_in)
            checkfull = prediction1[0,:,:,:]
            zy = checkfull.detach().numpy()
            blank_fl[:,x*128:(256+x*128), y*128:(256+y*128)] = zy
            gt3[countP] = zy
            countP += 1
            
    for channel in range(5): 
        countP = 0
        for P in range(49):
            a = gt3[P]
            gt2[P] = a[channel,:,:]
            if P<7:
                pass
            else:
                if P % 7 == 0:
                    countP += 1
                else:
                    ca = gt2[(P-8)] 
                    cb = gt2[(P-7)]
                    cc = gt2[(P-1)]
                    cd = gt2[(P)]
                    blank_fl[channel,countP*128:(countP+1)*128,
                                  (P - countP*7)*128:(P+1-countP*7)*128] = np.median([ca[128:,128:],cb[128:,0:128],
                                  cc[0:128,128:],cd[0:128,0:128]],axis=0)
        

        # Save the stitched predicted image                          
        im = blank_fl[channel,0:998,0:998]
        im1 = Image.fromarray(im)
        im1.save(f".../test/GAN/{channel}/{name_fluor}.tif")
        
        # Normalise and save the ground truth image for comparison!
        gtim3 = cv2.resize(gtim1[:,:,channel], dsize=(998,998), interpolation=cv2.INTER_CUBIC)
        mean, std = gtim3.mean(), gtim1.std()
        gtim = (gtim3-mean)/std
        gtim =  Image.fromarray(gtim)
        gtim.save(f".../test/GT/{channel}/{name_fluor}.tif")