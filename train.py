import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# TODO:
# - add per class accuracy
# - continuation 
# - better prints when training
# - saving checkpoints

class TrainingExperiment:
    def __init__(self, net, criterion, train_data:DataLoader, val_data:DataLoader=None):
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
        # stats
        self.epoch_loss = []
        self.epoch_acc = []
        self.epoch_count = 0

    def plot_loss(self, save_p:Path=None, figsize=(12,6)):
        vals = np.concatenate(self.epoch_loss) # concatenate different runs over the experiment
        plt.figure(figsize=figsize)
        plt.plot(range(1, len(list(vals))+1), vals) 
        plt.xlabel('epoch')
        plt.xticks(range(0,len(list(vals))+1,2))
        plt.title('loss')
        plt.legend(['train', 'validation'])
        if save_p is not None:
            plt.savefig(save_p)
        plt.show()

    def plot_acc(self, save_p:Path=None, figsize=(12,6)):
        vals = np.concatenate(self.epoch_acc) # concatenate different runs over the experiment 
        plt.figure(figsize=figsize)
        plt.plot(range(1, len(list(vals))+1), vals)
        plt.xlabel('epoch')
        plt.xticks(range(0,len(list(vals))+1,2))
        plt.title('accuracy')
        plt.legend(['train', 'validation'])
        if save_p is not None:
            plt.savefig(save_p)
        plt.show()

    def export_stats(self, out_dir:Path):
        ''' export '''
        out_json = {}

        out_json['epoch'] = self.epoch_count
        out_json['accuracy_history'] = np.concatenate(self.epoch_acc).tolist()
        out_json['train_class_occurences'] = self.train_class_occurences
        out_json['per class accuracy'] = {'train': self.per_class_acc['train'].cpu().numpy().tolist(), 
                                          'val': self.per_class_acc['val'].cpu().numpy().tolist()}
        with open(Path(out_dir) / 'train_stats.json', 'w') as f:
            json.dump(out_json, f)

        # also store figures
        self.plot_loss(save_p=Path(out_dir)/'loss.eps')
        self.plot_acc(save_p=Path(out_dir)/'accuracy.eps')

    def export_model(self, out_dir:Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        # export for inference only
        # torch.save(self.net, out_dir/f"{self.net.__class__.__name__}_epoch_{self.epoch_count}.pth")
        
        # # export for future usage here
        torch.save({
                    'epoch': self.epoch_count,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.epoch_loss[-1][-1],
                    }, 
                    out_dir/f"{self.net.__class__.__name__}_epoch_{self.epoch_count}.pth")

    def train(self, optimizer, num_epoch, scheduler=None):
        '''
        train
        args: 
            - optimizer
            - num_epoch: int
        '''
        self.optimizer = optimizer
        dataloaders = {'train': self.train_data, 'val':self.val_data}
        
        if isinstance(self.train_data.dataset, Subset):
            num_classes = self.train_data.dataset.dataset.get_num_classes()
        else:
            num_classes = self.train_data.dataset.get_num_classes()

        loss_hist = np.ones((num_epoch,2))*np.inf # [[train], [val]]
        acc_hist = np.zeros((num_epoch, 2))
        self.train_class_occurences = [0]*num_classes
        self.per_class_acc = {} # phase -> per_class accuracy

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        print(f"Starting training of {self.net}, using device {device}")

        for epoch in range(num_epoch):
            time_start = time.time()
            print(f"epoch: {self.epoch_count}")
            # iterate between train and val phase
            for phi, phase in enumerate(dataloaders):
                if dataloaders[phase] is None:
                    break 
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()
                
                # init per-epoch stats
                epoch_loss = 0.0
                correct_cls = 0.0
                correct_cls_per_class = {'train': torch.zeros(num_classes), 'val': torch.zeros(num_classes)}
                total_cls_per_class = {'train': torch.zeros(num_classes), 'val': torch.zeros(num_classes)}

                # tqdm around enumerate?
                for i, data_batch in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
                    inputs, labels = data_batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    # TODO: not getting batches form dataloaders. dataloaders not proper
                    optimizer.zero_grad()
                    out = self.net(inputs) 
                    # beware obtaining (out, embeddings)
                    if isinstance(out, tuple):
                        out = out[0]

                    # count train class occurences --distrubution
                    if epoch == 0:
                        for lbl in labels:
                            self.train_class_occurences[lbl] += 1
                        
                    loss = self.criterion(out, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item()
                                            
                    pred = torch.argmax(out, 1) # predicted class for each sample in batch
                    correct_cls += torch.sum(pred == labels).item()
                    for bidx, lbl in enumerate(labels):
                        if pred[bidx] == lbl:
                            correct_cls_per_class[phase][lbl] += 1
                        total_cls_per_class[phase][lbl] += 1
                assert correct_cls == correct_cls_per_class[phase].sum()

                loss_hist[epoch, phi] = epoch_loss  
                epoch_acc = correct_cls / len(dataloaders[phase].dataset)
                acc_hist[epoch, phi] = epoch_acc
                print(f"{phase.upper()} loss: {epoch_loss} | acc: {epoch_acc}")
            if scheduler:
                scheduler.step()
            self.epoch_count += 1

            self.per_class_acc['train'] = (correct_cls_per_class['train'] / total_cls_per_class['train'])
            self.per_class_acc['val'] = (correct_cls_per_class['val'] / total_cls_per_class['val'])
        
        self.epoch_loss.append(loss_hist)
        self.epoch_acc.append(acc_hist)
        print(f"training finished after {epoch+1} iterations")



if __name__=='__main__':
    from pathlib import Path
    import torch.optim as optim

    from models import *
    from dataset import build_dataset_out_of_dir_structure, get_data_loaders, visualize_sample, get_smaller_dataloader


    img_dir = 'traffic_signs_features/total_data_merged'
    RESIZE_TO = (128,128)
    BATCH_SIZE = 64

    imgs, labls, labls_2_id, cls_occurances = build_dataset_out_of_dir_structure(Path(img_dir))

    # dataloaders
    # train_data, test_data, val_data = get_data_loaders(imgs, labls, labls_2_id, 
    #                                                    batch_size=1, resize_to=resize_to)

    augmentations = None
    # quick overfit to a small subset of available dataset
    tiny_data = get_smaller_dataloader(800, imgs, labls, labls_2_id, batch_size=4, resize_to=RESIZE_TO)
    tiny_data_val = get_smaller_dataloader(400, imgs, labls, labls_2_id, batch_size=4, resize_to=RESIZE_TO)

    # train_data, test_data, val_data = get_data_loaders(imgs, labls, labls_2_id, 
    #                                                    batch_size=BATCH_SIZE, resize_to=RESIZE_TO, transform=augmentations)


    # net = Model1Vgg19(len(labls_2_id), img_size=resize_to)
    # net = ModelTinyHruz(num_out_classes=len(labls_2_id), img_size=resize_to)
    net = ResnetTiny(num_out_classes=len(labls_2_id), img_size=RESIZE_TO)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.002)

    tr = TrainingExperiment(net, loss, tiny_data, val_data=tiny_data_val)
    tr.train(optimizer, 1)
    tr.plot_loss()    
    tr.plot_acc()