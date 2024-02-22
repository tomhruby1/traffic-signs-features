import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

def run_inference(net, in_img_size, softmax_norm=False):
    '''Get last layer outputs given'''
    return


class Model1Vgg19(nn.Module):
    def __init__(self, num_out_classes, img_size=(224,224)):
        super().__init__()
        pretrained_model = timm.create_model('vgg19', pretrained=True) 
        # vgg19 backbone only (removed convMLP and classifier head)
        self.pretrained_backbone = (list(pretrained_model.children())[0])[0] # if input 224*224 output of this will be [64, 224, 224]
        self.fc_cls = nn.Linear(64*img_size[0]*img_size[1], num_out_classes)

    def forward(self, x):
        x = nn.functional.relu(self.pretrained_backbone(x))
        x = torch.flatten(x, start_dim=1) # flatten all dimensions except batch
        x = nn.functional.relu(self.fc_cls(x))

        return x

class ResnetTiny(nn.Module):
    # 128 x 128 --> 4 x 4 x 512
    # 224 x 224 --> 7 x 7 x 512
    def __init__(self, num_out_classes, img_size=(128,128)):
        super().__init__()
        pretrained_model = timm.create_model('resnet10t', pretrained=True)
        # remove the global pooling and last fc layer
        self.pretrained_backbone = torch.nn.Sequential(*(list(pretrained_model.children())[:-2]))
        self.fc = nn.Linear(4*4*512, num_out_classes)

    def forward(self, x):
        x = self.pretrained_backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        return x

class ModelTinyHruz(nn.Module):
    def __init__(self, num_out_classes=254, img_size=(224,224)):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3)
        self.pool1 = nn.MaxPool2d(2,1)
        self.norm1 = nn.BatchNorm2d(32) # 32 in_channels
        self.conv2 = nn.Conv2d(32,32,4) # stride=2
        self.pool2 = nn.MaxPool2d(2,1)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,3)
        self.pool3 = nn.MaxPool2d(2,1)
        # the ^^ goes makes the dimension of input image -10px smaller
        # e.g. (224*224 -> 214*214) (x 32 channels)
        self.fc4 = nn.Linear((img_size[0]-10)*(img_size[1]-10)*32, num_out_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = self.fc4(x)
        
        return x