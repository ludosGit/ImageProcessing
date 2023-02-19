import torch
import torch.nn as nn

########### LOSS FUNCTIONS ############

def eucl_distance(F1, F2):
    '''
    F1 F2: torch tensors
    '''
    d = torch.sqrt((((F1 - F2).pow(2))).sum(1))
    return d

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, F1, F2, dummy):
        '''
        F1, F2: embeddings learned for each of the input elements in the batch (pytorch tensors)
        the rows indicate the sample and the column the embedding's coordinates
        dummy: tensor of 0s and 1s, 1 if two images are similar 0 if dissimilar
        '''
        d = eucl_distance(F1, F2)
        losses = 0.5 * d.pow(2) * dummy + 0.5 * (1-dummy) * torch.clamp(self.margin-d, min=0)**2
        return losses.mean()

# def contrastive_loss(F1, F2, dummy, margin):
#     d = eucl_distance(F1, F2) # take the euclidean distance between the features
#     # d is a tensor
#     losses = 0.5 * d.pow(2) * dummy + 0.5 * (1-dummy) * torch.clamp(margin-d, min=0)**2 # takes the maximum between the value and min
#     return losses.mean()

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, F_anchor, F_positive, F_negative):
        '''
        F_anchor, F_positive, F_negative: embeddings learned for each of the input elements in the batch (pytorch tensors)
        the rows indicate the sample and the column the embedding's coordinates
        dummy: tensor of 0s and 1s, 1 if two images are similar 0 if dissimilar
        '''
        d_p = eucl_distance(F_anchor, F_positive)
        d_n = eucl_distance(F_anchor, F_negative)
        losses = torch.clamp(d_p-d_n + self.margin, min=0)
        return losses.mean()

# def triplet_loss(F_anchor, F_positive, F_negative, margin):
#     d_p = eucl_distance(F_anchor, F_positive)
#     d_n = eucl_distance(F_anchor, F_negative)
#     losses = torch.clamp(d_p-d_n + margin, min=0)
#     return losses.mean()

### Ranked List Loss
# Given an image xi, we aim to push its negative point farther than a boundary α and pull its positive one closer than
# another boundary α − m. Thus m is the margin between two boundaries.

# F1, F2 tensors of the features learned
def pairwise_margin_loss(F1, F2, dummy, alpha, m):
    d = eucl_distance(F1,F2)
    # push the positive points closer than alpha-m
    # push the negative further than alpha
    # thus m is the margin between two boundaries
    losses = dummy * torch.clamp(d-(alpha-m), min=0)+ (1 - dummy) * torch.clamp(alpha - d, min=0)
    return losses.mean()









