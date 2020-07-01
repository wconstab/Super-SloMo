
#[Super SloMo]
##High Quality Estimation of Multiple Intermediate Frames for Video Interpolation

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
import dataloader
from math import log10
import datetime
from tensorboardX import SummaryWriter
import random


random.seed(1337)
torch.manual_seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help='path to dataset folder containing train-test-validation folders')
parser.add_argument("--checkpoint_dir", type=str, required=True, help='path to folder for saving checkpoints')
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train. Default: 200.')
parser.add_argument("--train_batch_size", type=int, default=6, help='batch size for training. Default: 6.')
parser.add_argument("--init_learning_rate", type=float, default=0.0001, help='set initial learning rate. Default: 0.0001.')
parser.add_argument("--milestones", type=list, default=[100, 150], help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]')
parser.add_argument("--checkpoint_epoch", type=int, default=5, help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.')
parser.add_argument("--debug", type=str, default=None, help='dump model output')
parser.add_argument("--trace", action='store_true', default=False, help='trace model')
parser.add_argument("--script", action='store_true', default=False, help='script model')
args = parser.parse_args()

##[TensorboardX](https://github.com/lanpa/tensorboardX)
### For visualizing loss and interpolated frames


writer = SummaryWriter('log')


###Initialize flow computation and arbitrary-time flow interpolation CNNs.


assert torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



###Load Datasets


# Channel wise mean calculated on adobe240-fps training dataset
mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]
normalize = transforms.Normalize(mean=mean,
                                 std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

trainset = dataloader.SuperSloMo(root=args.dataset_root + '/train', transform=transform, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=False)

print(trainset)


###Create transform to display image from tensor


negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])


###Utils
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


###Loss and Optimizer


L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flowComp = model.UNet(6, 4).to(device)
        self.ArbTimeFlowIntrp = model.UNet(20, 5).to(device)
        self.trainFlowBackWarp = model.backWarp(352, 352, device)

    def forward(self, trainFrameIndex, I0, I1, IFrame):
        # Calculate flow between reference frames I0 and I1
        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
        
        # Extracting flows between I0 and I1 - F_0_1 and F_1_0
        F_0_1 = flowOut[:,:2,:,:]
        F_1_0 = flowOut[:,2:,:,:]
        
        fCoeff = model.getFlowCoeff(trainFrameIndex, I0.device)
        
        # Calculate intermediate flows
        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
        
        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0 = self.trainFlowBackWarp(I0, F_t_0)
        g_I1_F_t_1 = self.trainFlowBackWarp(I1, F_t_1)
        
        # Calculate optical flow residuals and visibility maps
        intrpOut = self.ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
        
        # Extract optical flow residuals and visibility maps
        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1   = 1 - V_t_0
        
        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0_f = self.trainFlowBackWarp(I0, F_t_0_f)
        g_I1_F_t_1_f = self.trainFlowBackWarp(I1, F_t_1_f)
        
        wCoeff = model.getWarpCoeff(trainFrameIndex, I0.device)
        
        # Calculate final intermediate frame 
        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

        # Loss
        recnLoss = L1_lossFn(Ft_p, IFrame)
            
        prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
        
        warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(self.trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(self.trainFlowBackWarp(I1, F_0_1), I0)
        
        loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
        loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
        loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
          
        # Total Loss - Coefficients 204 and 102 are used instead of 0.8 and 0.4
        # since the loss in paper is calculated for input pixels in range 0-255
        # and the input to our network is in range 0-1
        loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth

        return Ft_p, loss

the_model = Model()


optimizer = optim.Adam(the_model.parameters(), lr=args.init_learning_rate)
# scheduler to decrease learning rate by a factor of 10 at milestones.
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)


###Initializing VGG16 model for perceptual loss


vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3.to(device)
for param in vgg16_conv_4_3.parameters():
		param.requires_grad = False


### Initialization


dict1 = {'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}


### Training


import time

start = time.time()
cLoss   = dict1['loss']
valLoss = dict1['valLoss']
valPSNR = dict1['valPSNR']
checkpoint_counter = 0

if args.trace:
    for trainData, trainFrameIndex in trainloader:
        frame0, frameT, frame1 = trainData

        I0 = frame0.to(device)
        I1 = frame1.to(device)
        IFrame = frameT.to(device)

        the_model = torch.jit.trace(the_model, example_inputs=(trainFrameIndex, I0, I1, IFrame))
        break

if args.script:
    the_model = torch.jit.script(the_model)

### Main training loop
for epoch in range(dict1['epoch'] + 1, args.epochs):
    print("Epoch: ", epoch)
        
    # Append and reset
    cLoss.append([])
    valLoss.append([])
    valPSNR.append([])
    iLoss = 0
    
    # Increment scheduler count    
    scheduler.step()
    
    for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):
        
		## Getting the input and the target from the training set
        frame0, frameT, frame1 = trainData
        
        I0 = frame0.to(device)
        I1 = frame1.to(device)
        IFrame = frameT.to(device)
        
        optimizer.zero_grad()
        
        Ft_p, loss = the_model(trainFrameIndex, I0, I1, IFrame)
        if args.debug:
            torch.save(Ft_p, args.debug)
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        iLoss += loss.item()
