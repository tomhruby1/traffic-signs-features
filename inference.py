import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from pathlib import Path
import typing as T
import numpy as np
import json

# from .models import *


def run_inference(net:nn.Module, img:T.Union[torch.tensor, Path], 
                  in_img_size, softmax_norm=False):
    '''
    Get last layer outputs given model and 
    args:
        - in_img_size(height,width)
    '''
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
 
    net.to(device)

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
    
def get_prediction(img, model, classes_p='traffic_signs_features/total_data_CNN03/info.json'):
    
    with open(classes_p) as f:
        classes_data = json.load(f)
    
    id_2_label = classes_data['id_to_label']

    model.eval()
    out_vec = run_inference(model, img, (64,64), softmax_norm=True).cpu().detach().numpy()

    pred_label = id_2_label[np.argmax(out_vec)]

    return out_vec, f"{pred_label}: {np.max(out_vec)}%"

if __name__=='__main__':
    img_p = Path("traffic_signs_features/Cropped-Traffic-Signs-1obj-27_07_2023/B28/B28_7.jpg")
    model_p = "traffic_signs_features/resnet_tiny/ResnetTiny_epoch_6.pth"

    # model =  ModelTinyHruzBottleneck(bottleneck_size=128) # torch.load(str(model_p))
    # model.load_state_dict(checkpoint['model_state_dict'])
   
    model = torch.load(model_p)

    pred = get_prediction(img_p, model)

    print(pred[1])

    print("out vector sum:", np.sum(pred[0]))