import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import umap.umap_ as umap

data_frameGT = pd.read_csv('/.../FeatureData/GroundTruth_Features.csv.csv')
data_framepred = pd.read_csv('/.../FeatureData/Unet_Features.csv.csv')
data_framepredGAN = pd.read_csv('/.../FeatureData/cWGAN-GP_Features.csv.csv')


data_framecomb = pd.concat([data_frameGT, data_framepred, data_framepredGAN])
data_framecomb.head()
data_framecomb.Image_Metadata_Compound.value_counts()

reducer3 = umap.UMAP(min_dist=0.8)
data_framecomb = data_framecomb.loc[:, data_framecomb.columns != 'id']
data_frame3 = data_framecomb.loc[:, data_framecomb.columns != 'Image_Metadata_Compound']
data_frame3 = data_frame3.loc[:, data_frame3.columns != 'batch']
data_frame3 = data_frame3.loc[:, data_frame3.columns != 'gt0unet1gan2']
data_frame3 = data_frame3.loc[:, data_frame3.columns != 'CYTOPLASM_ImageNumber']
data_frame_data3 = data_frame3.values
scaled_data3 = StandardScaler().fit_transform(data_frame_data3)

embedding3 = reducer3.fit_transform(scaled_data3)
embedding3.shape

plt.ion()
plt.scatter(embedding3[:, 0], embedding3[:, 1],
    marker='o', s = 50, edgecolors='white', c=data_framecomb.Image_Metadata_Compound, cmap=ListedColormap(["tab:orange", "tab:cyan", "tab:olive"]),#color=["lightcoral","lightskyblue", "slategrey"], 
    )#fontsize=18)#label=['neg', 'pos', 'random'])
plt.gca().set_aspect('equal', 'datalim')
plt.xlabel("UMAP_1",fontsize=16)
plt.ylabel("UMAP_2",fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.title('Ground Truth UMAP: Treatment', fontsize=24)
red_patch = mpatches.Patch(color='tab:orange', label='Negative Control')
blue_patch = mpatches.Patch(color='tab:cyan', label='Positive Control')
green_patch = mpatches.Patch(color='tab:olive', label='Random Treatment')
plt.legend(handles=[red_patch, blue_patch, green_patch],fontsize=14)
plt.ioff()
plt.savefig('bothposnegGT.png')

plt.ion()
plt.scatter(embedding3[:, 0], embedding3[:, 1],
    marker='o', s = 50, edgecolors='white', c=data_framecomb.batch, cmap=ListedColormap(["tab:brown", "tab:blue"]),#color=["lightcoral","lightskyblue"], 
    )#fontsize=18)#'label=['batch1', 'batch2'])
plt.gca().set_aspect('equal', 'datalim')
plt.xlabel("UMAP_1",fontsize=16)
plt.ylabel("UMAP_2",fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.title('Ground Truth UMAP: Batch', fontsize=24)
red_patch = mpatches.Patch(color='tab:brown', label='Batch 1')
blue_patch = mpatches.Patch(color='tab:blue', label='Batch 2')
plt.legend(handles=[red_patch, blue_patch],fontsize=14)
plt.ioff()
plt.savefig('bothplateGT.png')


plt.ion()
plt.scatter(embedding3[:, 0], embedding3[:, 1],
    marker='o', s = 50, edgecolors='white', c=data_framecomb.gt0unet1gan2, cmap=ListedColormap(["tab:green", "tab:purple", "tab:red"]),#color=["lightcoral","lightskyblue", "slategrey"], 
    )#fontsize=18)#label=['neg', 'pos', 'random'])
plt.gca().set_aspect('equal', 'datalim')
plt.xlabel("UMAP_1", fontsize=16)
plt.ylabel("UMAP_2", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.title('Ground Truth UMAP: Treatment', fontsize=24)
red_patch = mpatches.Patch(color='tab:green', label='Ground Truth')
blue_patch = mpatches.Patch(color='tab:purple', label='U-Net')
green_patch = mpatches.Patch(color='tab:red', label='cWGAN-GP')
plt.legend(handles=[red_patch, blue_patch, green_patch],fontsize=14)
plt.ioff()
plt.savefig('bothunetganGT.png')
