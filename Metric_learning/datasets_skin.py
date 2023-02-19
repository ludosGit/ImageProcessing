import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import random
from torch.utils.data import DataLoader

############ DATASET CUSTOMIZED FOR SKINDATA ##############
# I NEED IT BEACUSE OF THE WAY I LOAD IT WITH IMAGEFOLDER: THE ELEMENTS ARE TUPLES OF LENGTH TWO
# WITH THE SECOND ELEMENT ALWAYS 0
# THIS IS BECAUSE DATALOADER EXPECTS THE DATA ORGANIZED IN DIFFERENT FOLDERS ACCORDING TO THE CLASS LABELS

def get_data_loader(train_data, test_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

########## CONTRASTIVE DATASET FOR SKIN CANCER DATA ###########

class ContrastiveDataset(Dataset):
    # INPUT:
    # tensor dataset
    def __init__(self, x_train, y_train, train = True, transform=None):
        self.is_train = train
        self.transform = transform  # indicates the transformation to apply to the dataset
        self.images = x_train
        self.labels = y_train
        self.index = np.array([i for i in range(len(x_train))])
        self.to_pil = transforms.ToPILImage() # only to be sure sure the transform applies

    def __len__(self):
        return len(self.images)

# main method of the class
    def __getitem__(self, item):
        anchor_img = self.images[item][0] # remember each image has label 0!!
        # if the dataset is in training mode, I want to return a couple of images
        # which can be either positive either negative with the SAME probability

        if self.is_train:
            anchor_label = self.labels[item]
            choice = random.random() # uniform distribution in [0,1)
            if choice < 0.5:
                positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]
                positive_item = random.choice(positive_list)
                positive_img = self.images[positive_item][0]
                dummy = 1
                if self.transform:
                    anchor_img = self.transform(self.to_pil(anchor_img))
                    positive_img = self.transform(self.to_pil(positive_img))
                return anchor_img, positive_img, dummy
            else:
                negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]
                negative_item = random.choice(negative_list)
                negative_img = self.images[negative_item][0]
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

########### TRIPLET DATASET FOR SKIN CANCER DATA ###########
# THE NEGATIVE SAMPLE IN THE TRIPLET IS SAMPLED UNIFORMLY AMONG THE CLASSES

class TripletDataset(Dataset):
    # INPUT:
    # x_train torch dataset
    # y_train numpy array!!
    def __init__(self, x_train, y_train, train = True, transform=None):
        self.is_train = train
        self.transform = transform  # indicates the transformation to apply to the dataset
        self.images = x_train
        self.labels = y_train
        self.index = np.array([i for i in range(len(x_train))])
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item): # returns a triplet this time, if in train mode
        anchor_img = self.images[item][0]

        # if the dataset is in training mode, I want to return a triplet of images

        if self.is_train:
            anchor_label = self.labels[item]
            positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]
            positive_index = random.choice(positive_list)
            positive_img = self.images[positive_index][0]

            # list if indexes of the negative labels
            labels_unique = np.unique(self.labels)
            negative_labels = labels_unique[labels_unique != anchor_label]
            # sample a label among the negative ones: in this way I am sure avery class is taken with same probability
            negative_label = random.choice(negative_labels)
            negative_list = self.index[self.labels == negative_label]
            # after select a random image in the class
            negative_index = random.choice(negative_list)
            negative_img = self.images[negative_index][0]

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

