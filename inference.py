import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from pathlib import Path
import typing as T
import numpy as np

from models import *

def run_inference(net:nn.Module, img:T.Union[torch.tensor, Path], 
                  in_img_size, softmax_norm=False):
    '''
    Get last layer outputs given model and 
    args:
        - in_img_size(height,width)
    '''
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
 
    if isinstance(img, Path):
        img = read_image(str(img)).float()
 
    # add dummy batch dim TODO: batching?
    img = img.unsqueeze(0)
    img = TF.resize(img, in_img_size).to(device)

    # normalize as done during training
    img = TF.normalize(img, (0.0,0.0,0.0), (255,255,255))
    
    out = net(img)
    out = out.squeeze(0)
    
    if softmax_norm:
        return F.softmax(out)
    else:
        return out    
    
def get_prediction(img, model_p):
    model = torch.load(str(model_p))
    model.eval()
    out_vec = run_inference(model, img, (128,128), softmax_norm=True).cpu().detach().numpy()

    return out_vec

if __name__=='__main__':
    img_p = Path("Cropped-Traffic-Signs-1obj-27_07_2023/B28/B28_7.jpg")
    model_p = "resnet-tiny/ResnetTiny_epoch_6.pth"
    
    out = get_prediction(img_p, model_p)

    print(out)

    print("out vector sum:", np.sum(out))