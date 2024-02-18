import torch 
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO:
# - add accuracy (correct_cls/dataset_size)
# - continuation 
# - better prints when training
# - saving checkpoints

class TrainingExperiment:
    def __init__(self, net, criterion, train_data, val_data=None):
        '''
        args:
            - criterion: loss fcn
            - train_data: training dataset dataloader
            - val_data: validation set dataloader:
        '''
        self.net = net
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        self.epoch_loss = []

    def plot_loss(self):
        plt.plot(np.concatenate(self.epoch_loss)) # concatenate different runs over the experiment
        plt.xlabel('epoch')
        plt.title('loss')
        plt.legend(['train', 'validation'])
        plt.show()

    def train(self, optimizer, num_epoch):
        '''
        train
        args: 
            - optimizer
            - num_epoch: int
        '''
        dataloaders = {'train': self.train_data, 'val':self.val_data}
        loss_hist = np.ones((num_epoch,2))*np.inf
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        print(f"Starting training of {self.net}, using device {device}")

        for epoch in range(num_epoch):
            time_start = time.time()
            print(f"epoch: {epoch}")
            # iterate between train and val phase
            for phi, phase in enumerate(dataloaders):
                if dataloaders[phase] is None:
                    break 
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()
                
                epoch_loss = 0.0

                # tqdm around enumerate?
                for i, data_batch in tqdm(enumerate(dataloaders[phase])):
                    inputs, labels = data_batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    # TODO: not getting batches form dataloaders. dataloaders not proper
                    optimizer.zero_grad()
                    out = self.net(inputs)
                    loss = self.criterion(out, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # else: #get accuracy on validation
                    #     pred = torch.argmax(out,1)
                    #     acc = torch.sum(pred == labels)
                    #     val_acc += acc.item() 
                        
                    epoch_loss += loss.item()
                loss_hist[epoch, phi] = epoch_loss  
                print(f"{phase} loss: {epoch_loss}")
        self.epoch_loss.append(loss_hist)
        print(f"training finished after {epoch+1} iterations")

from pathlib import Path
from models import *
from dataset import build_dataset_out_of_dir_structure, get_data_loaders, visualize_sample, get_smaller_dataloader
import torch.optim as optim

if __name__=='__main__':
    img_dir = 'total_data_CNN03'
    resize_to = (128,128)

    imgs, labls, labls_2_id = build_dataset_out_of_dir_structure(Path(img_dir))

    # dataloaders
    # train_data, test_data, val_data = get_data_loaders(imgs, labls, labls_2_id, 
    #                                                    batch_size=1, resize_to=resize_to)

    # quick overfit to a small subset of available dataset
    tiny_data = get_smaller_dataloader(800, imgs, labls, labls_2_id, batch_size=4, resize_to=resize_to)
    tiny_data_val = get_smaller_dataloader(400, imgs, labls, labls_2_id, batch_size=4, resize_to=resize_to)

    # net = Model1Vgg19(len(labls_2_id), img_size=resize_to)
    net = ModelTinyHruz(num_out_classes=len(labls_2_id), img_size=resize_to)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.002)

    tr = TrainingExperiment(net, loss, tiny_data, val_data=tiny_data_val)
    tr.train(optimizer, 3)
    tr.plot_loss()    