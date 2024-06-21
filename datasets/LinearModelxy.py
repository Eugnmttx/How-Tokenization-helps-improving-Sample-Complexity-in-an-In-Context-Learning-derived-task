from itertools import *

import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LinearModelxy(Dataset):

    def __init__(
            self,
            train_size,
            num_samples,
            dimension,
            seed_function=1,
            seed_sample=1,
            test_size=0,
            transform=None,
    ):
        
        torch.manual_seed(seed_sample)
        samples = torch.randn(train_size+test_size,num_samples+1,dimension)
        samples = samples/torch.norm(samples,dim=2,keepdim=True)

        torch.manual_seed(seed_function)
        functions = torch.randn(train_size+test_size,dimension)

        samples_labels = torch.einsum('ik,ijk->ij',functions,samples)

        labels = torch.einsum('ij,ijk,ik->i',samples_labels[:,:num_samples],samples[:,:num_samples,:],samples[:,-1,:])/num_samples

        samples_labels[:,-1] = torch.zeros(train_size+test_size)

        samples = torch.cat([samples,samples_labels.view(train_size+test_size,num_samples+1,1)],dim=2)

        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
        	idx: sample index

        Returns:
            Feature-label pairs at index            
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y
