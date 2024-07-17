import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
# TODO: flatten the output from the backbone and output as embedding
class ResnetTiny(nn.Module):
    # 224 x 224 --> 7 x 7 x 512
    # 128 x 128 --> 4 x 4 x 512
    def __init__(self, num_out_classes=254, img_size=(128,128)):
        super().__init__()
        pretrained_model = timm.create_model('resnet10t', pretrained=True)
        # remove the global pooling and last fc layer
        self.pretrained_backbone = torch.nn.Sequential(*(list(pretrained_model.children())[:-2]))
        self.fc = nn.Linear(4*4*512, num_out_classes)

    def forward(self, x):
        emb = self.get_embedding(x)
        x = self.fc(emb)
        
        return x, emb
    
    def get_embedding(self, x):
        x = self.pretrained_backbone(x)
        x = torch.flatten(x, start_dim=1)

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
        self.pool3 = nn.MaxPool2d(2,2)
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
    
class ModelTinyHruzBottleneck(nn.Module):
    def __init__(self, num_out_classes=254, img_size=(64,64), bottleneck_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3)
        self.pool1 = nn.MaxPool2d(2,1)
        self.norm1 = nn.BatchNorm2d(32) # 32 in_channels
        self.conv2 = nn.Conv2d(32,32,4) # stride=2
        self.pool2 = nn.MaxPool2d(2,1)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,3)
        self.pool3 = nn.MaxPool2d(2,2)
        # 59 x 59 for 128x128 input
        # 19x19 for 48x48 input
        # TODO: ^^^ this is dump -- calculate properly
        self.fc4 = nn.Linear(59*59*32, bottleneck_size)
        self.fc5 = nn.Linear(bottleneck_size, num_out_classes)
    
    def forward(self, x):
        embedding = self.get_deep_feature_vector(x)
        x = F.relu(self.fc5(embedding))

        return x, embedding
    
    def get_deep_feature_vector(self, x):
        '''run forward pass and return the output of the second to last fc layer'''
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc4(x))

        return x
    
class LeNet(nn.Module):
    def __init__(self, num_out_classes, img_size=(28,28)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.pool = nn.AvgPool2d(2, 2) # out: [6, 14, 14]
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_out_classes)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.pool(x)
        x = F.sigmoid(self.conv2(x))
        x = self.pool(x)
        emb = torch.flatten(x,1)
        x = F.sigmoid(self.fc1(emb))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x, emb
    
class LeNetRGB(nn.Module):
    def __init__(self, num_out_classes, img_size=(28,28)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1, padding=2)
        self.pool = nn.AvgPool2d(2, 2) # out: [6, 14, 14]
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear((img_size[0]-23)*(img_size[1]-23)*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_out_classes)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.pool(x)
        x = F.sigmoid(self.conv2(x))
        x = self.pool(x)
        emb = torch.flatten(x,1)
        x = F.sigmoid(self.fc1(emb))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        return x, emb