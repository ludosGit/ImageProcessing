from pathlib import Path
import requests
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import random
from torch.utils.data import DataLoader, TensorDataset

##### DOWNLOAD MNIST DATA #######

def download_MNIST():
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    import pickle
    import gzip

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_test, y_test), _) = pickle.load(f, encoding="latin-1")
    return x_train, y_train, x_test, y_test


# Covert the data to pytorch tensors and make a TensorDataset

def to_torch_data(x_train, y_train, x_test, y_test):
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)
    # merge the tensors in a TensorDataset
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)
    return train_data, test_data


####### CUSTOMIZED TORCH DATASET FOR CONTRASTIVE LOSS AND MNIST DATASET #########

class ContrastiveDataset(Dataset):
    # INPUT:
    # tensor dataset
    def __init__(self, data, train = True, transform=None, target_transform=None):
        self.is_train = train
        self.transform = transform  # indicates the transformation to apply to the dataset
        self.target_transform = target_transform
        x, y = data[:] # saparate observation from labels, thanks to the way we merged them at the beginning
        # define the attributes as numpy arrays
        self.images = x.cpu().detach().numpy()
        self.labels = y.cpu().detach().numpy()
        self.index = np.array([i for i in range(len(x))])
        self.to_pil = transforms.ToPILImage() # only to be sure sure the transform applies

    def __len__(self):
        return len(self.images)

# main method of the class
    def __getitem__(self, item):
        anchor_img = self.images[item].reshape(28, 28)
        # if the dataset is in training mode, I want to return a couple of images
        # which can be either positive either negative with the SAME probability

        if self.is_train:
            anchor_label = self.labels[item]
            choice = random.random() # uniform distribution in [0,1)
            if choice < 0.5:
                positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]
                positive_item = random.choice(positive_list)
                positive_img = self.images[positive_item].reshape(28, 28)
                dummy = 1
                if self.transform:
                    anchor_img = self.transform(self.to_pil(anchor_img))
                    positive_img = self.transform(self.to_pil(positive_img))
                return anchor_img, positive_img, dummy
            else:
                negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]
                negative_item = random.choice(negative_list)
                negative_img = self.images[negative_item].reshape(28, 28)
                dummy = 0
                if self.transform:
                    anchor_img = self.transform(self.to_pil(anchor_img))
                    negative_img = self.transform(self.to_pil(negative_img))
                return anchor_img, negative_img, dummy
        else:
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
            return anchor_img

    def set_train(self, bool):
        self.is_train = bool
        return None


###### GET TRAIN AND TEST DATALOADERS

def get_data_loader(train_data, test_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


######## DATASET CLASS FOR TRIPLET LOSS AND MNIST DATASET ###########

class TripletDataset(Dataset):
    # INPUT:
    # data tensor dataset
    def __init__(self, data, train = True, transform=None):
        self.is_train = train
        self.transform = transform  # indicates the transformation to apply to the dataset
        x, y = data[:]
        self.images = x
        self.labels = y
        self.index = np.array([i for i in range(len(x))])
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item): # returns a triplet this time, if in train mode
        anchor_img = self.images[item].reshape(28, 28)

        # if the dataset is in training mode, I want to return a triplet of images

        if self.is_train:
            anchor_label = self.labels[item]
            positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]
            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item].reshape(28, 28)

            negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_img = self.images[negative_item].reshape(28, 28)
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
                positive_img = self.transform(self.to_pil(positive_img))
                negative_img = self.transform(self.to_pil(negative_img))
            return anchor_img, positive_img, negative_img
        else:
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
            return anchor_img

    def set_train(self, bool):
        self.is_train = bool
        return None


######## DATASET CLASS FOR TRIPLET LOSS WITH NEGATIVE LABELS SAMPLED UNIFORMLY AMONG THE CLASSES #############
class UniformTripletDataset(Dataset):
    # INPUT:
    # data tensor dataset
    def __init__(self, data, train = True, transform=None):
        self.is_train = train
        self.transform = transform  # indicates the transformation to apply to the dataset
        x, y = data[:]
        self.images = x
        self.labels = y
        self.index = np.array([i for i in range(len(x))])
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item): # returns a triplet this time, if in train mode
        anchor_img = self.images[item].reshape(28, 28)
        # if the dataset is in training mode, I want to return a triplet of images
        if self.is_train:
            anchor_label = self.labels[item]
            positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]
            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item].reshape(28, 28)
            # list if indexes of the negative labels
            labels_unique = torch.unique(self.labels)
            negative_labels = labels_unique[labels_unique != anchor_label]
            # sample a label among the negative ones: in this way I am sure avery class is taken with same probability
            negative_label = random.choice(negative_labels)
            negative_list = self.index[self.labels == negative_label]
            # after select a random image in the class
            negative_index = random.choice(negative_list)
            negative_img = self.images[negative_index].reshape(28, 28)
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
                positive_img = self.transform(self.to_pil(positive_img))
                negative_img = self.transform(self.to_pil(negative_img))
            return anchor_img, positive_img, negative_img
        else:
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
            return anchor_img

    def set_train(self, bool):
        self.is_train = bool
        return None


######## ATTEMPT TO DEFINE DATASET CLASS FOR RANKED LIST LOSS (NOT IMPLEMENTED IN THE PAPER)

class RankedList(Dataset):
    # INPUT:
    # data tensor dataset
    def __init__(self, data, size, train = True, transform=None):
        self.is_train = train
        self.transform = transform  # indicates the transformation to apply to the dataset
        x, y = data[:]
        self.images = x
        self.labels = y
        self.index = np.array([i for i in range(len(x))])
        self.to_pil = transforms.ToPILImage()
        self.size = size  # size of class with most elements

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item): # returns a triplet this time, if in train mode
        anchor_img = self.images[item].reshape(28, 28)

        # if the dataset is in training mode, I want to return the image together with its positive and negative sets

        if self.is_train:
            anchor_label = self.labels[item]
            positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]
            negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]
            positive_images = self.images[positive_list]
            negative_images = self.images[negative_list]

            # Non-trivial Sample Mining will be in the training loop
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))

            return anchor_img, item
        else:
            if self.transform:
                anchor_img = self.transform(self.to_pil(anchor_img))
            return anchor_img

    def set_train(self, bool):
        self.is_train = bool
        return None


