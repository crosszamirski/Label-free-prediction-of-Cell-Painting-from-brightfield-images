import torch
from torch import nn

class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()
        self.mae_loss = nn.L1Loss(reduce = False)
        
    def forward(self, out_images, target_images, epoch):
        image_loss = self.mae_loss(out_images, target_images)
        #image_loss.requires_grad = True
        #print(image_loss.mean())
        combined_loss = image_loss.mean() 
        #combined_loss.requires_grad = True
        return combined_loss

if __name__ == "__main__":
    g_loss = GenLoss()
    #print(g_loss)