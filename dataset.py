### PYTORCH DATASET AND DATALOADER HERE
import typing as T
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transf
import matplotlib.pyplot as plt
import json

# TODO: standardize --normalize with std and mean of the whole dataset

CLASSES_FILE = 'info.json' # inside the dataset dir

class Dataset(Dataset):
    def __init__(self, images:tuple, labels:tuple, id_to_label:tuple, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.id_to_label = id_to_label

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        '''returns image as a tensor and corresponding label'''
        # images expected already loaded in memory
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        # if self.target_transform:
        #     label = self.target_transform(label)
        label_id = self.id_to_label.index(self.labels[idx])
        return img, torch.tensor(label_id)
    
    def get_item_of_label(self, label):
        ''' get a first sample of a training pair given label '''
        idx = self.labels.index(label)
        return self.__getitem__(idx)

    def get_label(self, label_id):
        '''get category string label given it's id'''
        return self.id_to_label[label_id.item()]
    
    def get_num_classes(self):
        return len(self.id_to_label)




# THIS applies transforms to all 3 dataloaders
def get_data_loaders(imgs, labels, id_to_label, transform:list=None, train_size=0.8, 
                     test_size=0.1, batch_size=16, resize_to=(224,224)) -> T.Tuple[DataLoader, DataLoader, DataLoader]:
    '''Get (train, test, val) dataloaders'''
    # fix random for reproducible split
    gen = torch.Generator()
    gen.manual_seed(123)
    torch.random.manual_seed(123) # this one is needed! 

    
    # normalization to 0-1
    transf_base = [
        transf.ConvertImageDtype(torch.float32), # this should also map it to (0-1)...
        # transf.RandomHorizontalFlip(p=0.5),
        # transf.RandomCrop(size=(224,224)) # TODO: not sure about random crop or resize
        transf.Resize(resize_to)
    ]
    if transform is not None:
        for t in transform:
            transf_base.append(t)
    transform = transf.Compose(transf_base)
    
    
    dataset = Dataset(imgs, labels, id_to_label, transform=transform)

    train_len = int(np.round(dataset.__len__() * train_size))
    test_len = int(np.round(dataset.__len__() * test_size))
    val_len = dataset.__len__() - train_len - test_len
    print(f"data split; train:{train_len}, test:{test_len}, val:{val_len}")


    return tuple([DataLoader(data, batch_size=batch_size) 
                  for data in random_split(dataset, [train_len, test_len, val_len])])


def get_smaller_dataloader(size, imgs, labels, id_to_label, transform=None, 
                           batch_size=16, resize_to=(224,224)):
    '''returns dataloader with subset of size size of the dataset'''
    # fix random for reproducible split
    gen = torch.Generator()
    gen.manual_seed(123)
    torch.random.manual_seed(123) # this one is needed!

    # normalization to 0-1
    transf_base = [
        transf.ConvertImageDtype(torch.float32),
        # transf.Normalize((0.0,0.0,0.0), (255,255,255)),
        # transf.RandomHorizontalFlip(p=0.5),
        # transf.RandomCrop(size=(224,224)) # TODO: not sure about random crop or resize
        transf.Resize(resize_to)
    ]
    if transform is not None:
        for t in transform:
            transf_base.append(t)
    transform = transf.Compose(transf_base)
    
    
    dataset = Dataset(imgs, labels, id_to_label, transform=transform)
    small_dataset, _ = random_split(dataset, [size, dataset.__len__()-size])
    
    return DataLoader(small_dataset, batch_size=batch_size)

def build_dataset_out_of_dir_structure(parent_dir:Path, expected_labels:list=None):
    '''
    iterates over directories named as labels returns list of images and label

    returns: (images, labels, labels_to_id, label_counts)
    '''
    IMG_SUFFIXES = ["jpg","png"]
    
    images = []
    labels = [] # lables[i] : label for image images[i]
    labels_to_id = [] # category id -> string label mapping
    label_counts = {}
    for dir in parent_dir.iterdir():
        if expected_labels and dir.name not in expected_labels:
            # assert dir.name in expected_labels
            print(f"{dir.name} not found in expected labels")
        else:
            print(f"{dir.name} loaded")
        if dir.name not in labels_to_id:
            label_counts[dir.name] = 0
        for img_p in dir.rglob('*'):
            if img_p.name.split(".")[-1] not in IMG_SUFFIXES:
                print("unexpected image suffix:", img_p.name.split(".")[-1])
            try:
                img = read_image(str(img_p), mode=ImageReadMode.RGB) #  mode=ImageReadMode.UNCHANGED
                if img.shape[0] != 3:
                    raise Exception(f"error 3 channels img expected,instead got: {img.shape}")
                else:
                    images.append(img) # img.float() -- this makes it ugly
                    labels.append(dir.name)
                    label_counts[dir.name] += 1
            except Exception as ex:
                print(f"loading {img_p} failed:\n{ex}")
    print(f"Dataset of {len(images)} images loaded")

    try:
        with open(parent_dir/CLASSES_FILE, 'r') as f:
            classes_dict = json.load(f)
    except Exception as ex:
        print(f"could not find {parent_dir/CLASSES_FILE}")
        print("looking into the parent dir")
        with open(parent_dir.parent/CLASSES_FILE) as f:
            classes_dict = json.load(f)
        print("loaded")

    return tuple(images), tuple(labels), tuple(classes_dict['id_to_label']), label_counts

def build_per_class_datasets_out_of_dir_structure(parent_dir:Path, id_2_label):
    ''' returns dict: label->torch.Dataset'''
    
    IMG_SUFFIXES = ["jpg","png"]
    datasets = {}
    for dir in parent_dir.iterdir():
        images = []
        labels = []
        for img_p in dir.rglob('*'):
            if img_p.name.split(".")[-1] not in IMG_SUFFIXES:
                print("unexpected image suffix:", img_p.name.split(".")[-1])
            try:
                img = read_image(str(img_p), mode=ImageReadMode.RGB) #  mode=ImageReadMode.UNCHANGED
                if img.shape[0] != 3:
                    raise Exception(f"error 3 channels img expected,instead got: {img.shape}")
                else:
                    images.append(img)
                    labels.append(dir.name)
            except Exception as ex:
                print(f"loading {img_p} failed:\n{ex}")
        datasets[dir.name] = Dataset(images, labels, id_2_label, transform=None)
        print(f"dataset of {dir.name} built")
    
    return datasets

def split_per_class_dataset(dataset, train_size=0.8, test_size=0.1) -> T.Tuple[Dataset,Dataset,Dataset]:
    ''' returns (train, val, test) tuple of datasets'''
    gen = torch.Generator()
    gen.manual_seed(123)
    torch.random.manual_seed(123) # this one is needed!

    train_len = int(np.round(dataset.__len__() * train_size))
    test_len = int(np.round(dataset.__len__() * test_size))
    val_len = dataset.__len__() - train_len - test_len
    print(f"split; train:{train_len}, val:{val_len}, test:{test_len}, ")

    return random_split(dataset, [train_len, val_len, test_len])


def serialize_per_class_datasets(datasets:dict, target_parent_dir:Path):
    ''' store dictionary of label->torch.Dataset to dir structure'''
    from torchvision.transforms.functional import to_pil_image
    # TODO: before saving each dataset do a split + optionally to train add transform
    for label, dataset in datasets.items():
        (target_parent_dir/label).mkdir()
        for idx in range(len(dataset)):
            img, img_label = dataset.__getitem__(idx)
            assert dataset.dataset.id_to_label[img_label.item()] == label
            img_pil = to_pil_image(img)
            img_pil.save(target_parent_dir/label/f"{idx}.jpg")

def visualize_sample(dataloader):
    ''''''
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    dt = iter(dataloader.dataset)
    for i in range(1, cols * rows + 1):
        img, label = next(dt)
        img *= 255 # unnormalize
        img = img.permute(1,2,0).int()
        figure.add_subplot(rows, cols, i)
        # first dataset is the subset second is the reference to the main one...
        plt.title(dataloader.dataset.dataset.get_label(label))
        plt.axis("off")
        plt.imshow(img.cpu().numpy())
    plt.show()


if __name__=='__main__':

    # this vvv 
    
    IMG_DIR = 'traffic_signs_features/total_data_merged_filt'

    # load dataset to memory
    imgs, labls, labls_2_id, cls_occurances = build_dataset_out_of_dir_structure(Path(IMG_DIR))

    TARGET_PARENT_DIR = Path('traffic_signs_features/new_dataset')

    datasets = build_per_class_datasets_out_of_dir_structure(Path(IMG_DIR), labls_2_id)

    datasets_train, datasets_val, datasets_test = {}, {}, {}
    for label, dataset in datasets.items():
        datasets_train[label], datasets_val[label], datasets_test[label] = split_per_class_dataset(dataset)


    for data_section in [('train', datasets_train), ('val', datasets_val), ('test', datasets_test)]:
        (TARGET_PARENT_DIR/data_section[0]).mkdir(parents=True)
        serialize_per_class_datasets(data_section[1], TARGET_PARENT_DIR/data_section[0])