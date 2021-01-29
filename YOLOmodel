import torch
import torch.nn as nn

core = {'layer1': [(64, 7, 3, 2)], 
 'layer2': [(192, 3, 1, 1)],
 'layer3': [(128, 1, 0, 1), (256, 3, 1, 1), (256, 1, 0, 1), (512, 3, 1, 1)],
 'layer4': [(256, 1, 0, 1), (512, 3, 1, 1)]*4 + [(512, 1, 0, 1), (1024, 3, 1, 1)],
 'layer5': [(512, 1, 0, 1), (1024, 3, 1, 1)]*2 + [(1024, 3, 1, 1), (1024, 3, 1, 2)] + [(1024, 3, 1, 1)]*2
}
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ConvolutionBlock, self).__ini__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyRelu(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class YOLO(nn.Module):
    def __init__(self, image_channels):
        super(YOLO, self).__init__()
        self.in_channel = image_channels
        self.core = self.create_core()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(1024 * 7 * 7, 4096), 
                                nn.LeakyReLU(0.1), nn.Linear(4096, 7 * 7 * (20 + 2*5)))
    def forward(self, image):
        x = self.core(image)
        return self.fc(x)
    
    def create_core(self):
        layers = []
        for layer in core:
            for conv in core[layer]:
                layers.append(
                    ConvBlock(self.in_channels, conv[0], conv[1], conv[2], conv[3])
                )
                self.in_channels = conv[0]
            if layer != 'layer5':
                layers.append(nn.MaxPool2d(2, 2))
        
        return nn.Sequential(*layers)
    


class YOLOLoss(nn.Module):
    def __init__(self, n_Classes=20, split=7, n_BBs=2):
        super(YOLOLoss, self).__init__()
        self.split = split
        self.n_BBs = n_BBs
        self.n_class = n_Classes
        self.lambadnoobj = 0.5
        self.lambacoord = 5
        self.mse = nn.MSELoss(reduction='sum')
       
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.split, self.split, self.n_class + self.n_BBs * 5)
        
        b1iou = calculate_iou(predictions[..., 21:25], target[..., 21:25])
        b2iou = calculate_iou(predictions[..., 26:30], target[..., 21:25])
        ious = torch.stack([b1iou, b2iou])
        iouValue, boxIndex = torch.max(ious, dim=0)
        exists_box = target[..., 20:21]
        loss = 0
        
        predictedBB = exists_box*(boxIndex * predictions[..., 26:30] + (1 - boxIndex) * predictions[..., 21:25])
        targetbox = exists_box * target[..., 21:25]
        ########### loss attributed to coordinates#################
        predictedBB[..., 2:4] = torch.sign(predictedBB[..., 2:4]) * torch.sqrt(torch.abs(predictedBB[..., 2:4]) + 1e-7)
        targetbox[..., 2:4] = torch.sqrt(targetbox[..., 2:4])
        coordsLoss = self.mse(torch.flatten(predictedBB, end_dim=-2), torch.flatten(targetbox, end_dim=-2))
        ############################################################
        ########## loss attributed to detection of object in a box ##################### 
        existProb = exists_box*(boxIndex * predictions[..., 25:26] + (1 - boxIndex) * predictions[..., 20:21])
        objsloss = self.mse( torch.flatten(existProb), torch.flatten(target[..., 20:21]))
        ################################################################################
        
        ########## loss attributed to mistakenly predicting no object in a box #########
        noobjsloss = self.mse(torch.flatten((1 - exists_box)*predictions[..., 20:21], start_dim=1),
                              torch.flatten((1 - exists_box)*target[..., 20:21], start_dim=1))
        noobjsloss += self.mse(torch.flatten((1 - exists_box)*predictions[..., 25:26], start_dim=1),
                               torch.flatten((1 - exists_box)*target[..., 20:21]))
        #################################################################################
        
        ########## loss attributed to prediction of the class ###########################
        classloss = torch.mse(torch.flatten(exists_box*predictions[..., 0:20], end_dim=-2),
                              torch.flatten(exists_box*target[..., 0:20], end_dim=-2))
        #################################################################################
        
        loss = self.lambdacoord * coordsloss + objsloss + self.lambdanoobj*noobjsloss + classloss
        return loss
