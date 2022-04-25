import torch
from torch import nn
#from torchvision.models.vgg import vgg16

#option for perception loss - not used

class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()
        #self.mse_loss = nn.MSELoss(reduce = False)
        self.mae_loss = nn.L1Loss(reduce = False)
        
    def forward(self, out_labels, out_images, target_images, epoch):
        # Adversarial Loss
        adversarial_loss = -torch.mean(out_labels)
        #adversarial_loss.requires_grad = True
        
        # Perception Loss
        #perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        
        # Image Loss
        image_loss = self.mae_loss(out_images, target_images)
        #image_loss.requires_grad = True

        combined_loss = image_loss.mean() + 0.01 * adversarial_loss/(epoch+1) #+ 0.006 * perception_loss
        return combined_loss



if __name__ == "__main__":
    g_loss = GenLoss()
    #print(g_loss)