
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math as m
from sys import argv
import argparse

import models
import datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

# Parsing for parameters
parser = argparse.ArgumentParser()

parser.add_argument("-b", "--bias", type=bool,
                    help="Set the bias to True or False")
parser.add_argument("-M", "--num_samples", type=int,
                    help="Choose the number of examples/samples in an el. of the train. set")
parser.add_argument("-d", "--dimension", type=int,
                    help="Choose the dimension of the input.")
parser.add_argument("-bs", "--batch_size", type=int,
                    help="Choose the  batch size for SGD.")
parser.add_argument("-plr","--pow_lr", type=float,
                    help="select the learning rate for the training")
parser.add_argument("-T","--num_epochs", type=int,
                    help="Choose the number of epochs for the training")
parser.add_argument("-dt","--hidden_dim_token",type=int,
                    help="Choose the width of the hidden layers")
parser.add_argument("-df","--hidden_dim_feature",type=int,
                    help="Choose the width of the hidden layers")
parser.add_argument("-f","--frequency",type=int,
                    help="Choose the frequency to store the losses")
parser.add_argument("-sd","--seed_function", type=int,
                    help="Choose the seed for initialising the data function")
parser.add_argument("-sds","--seed_sample", type=int,
                    help="Choose the seed for initialising the data samples")
parser.add_argument("-sdm","--seed_model", type=int,
                    help="Choose the seed for initialising the model")
parser.add_argument("-es","--error_save", type=bool,
                    help="Choose to save the error or not")
parser.add_argument("-ms","--model_save", type=bool,
                    help="Choose to save the model or not")
parser.add_argument("-P","--training_size", type=int,
                    help="Choose the size of the training set")

args = parser.parse_args()

print('ok1')

# Parameters for the dataset
Ps = [10, 100, 300, 500, 750, 1000, 1250, 1500, 2000, 3000, 4500, 6000, 8000, 10000, 12500, 15000, 20000, 30000, 40000, 50000, 75000, 100000] #22
P = Ps[args.training_size]
M = args.num_samples
d = args.dimension
bs = args.batch_size
test_size = P
num_batch_train = m.ceil(P/bs)
num_batch_test = m.ceil(test_size/bs)

# Dataset

title = 'XY model'
dataset = datasets.LinearModelxy(
    train_size = P,
    num_samples = M,
    dimension = d,
    seed_function=args.seed_function,
    seed_sample=args.seed_sample,
    test_size=test_size,
    transform=None
    )

dataset.samples, dataset.labels = dataset.samples.to(device), dataset.labels.to(device)

train_dataset = TensorDataset(dataset.samples[:P], dataset.labels[:P])
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)

test_dataset = TensorDataset(dataset.samples[P:], dataset.labels[P:])
test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

# Initialisation of the model
torch.manual_seed(args.seed_model)

model = models.MLP_mixer(
    dim_token = d+1,
    dim_feature = M+1,
    hidden_dim_token = args.hidden_dim_token,
    hidden_dim_feature = args.hidden_dim_feature,
    out_dim = 1,
    norm = 'mf'
)

model = model.to(device)
print('ok3')

# Parameters for the training
eta = (args.hidden_dim_token*args.hidden_dim_feature)**args.pow_lr
T = int(args.num_epochs)

# Variables for the result analysis
train_error = np.zeros(T)
test_error = np.zeros(T)
freq = args.frequency

# Parameters
param_str = 'M= '+str(M)+' d= '+str(d)+' P= '+str(P)+' hdim_token= '+str(args.hidden_dim_token)+' hdim_feature= '+str(args.hidden_dim_feature)

# Variables for the training
optimiser = optim.SGD(model.parameters(), eta)
loss_fct = nn.MSELoss(reduction='mean')

for i in range(T):
    optimiser.zero_grad()
    for input, label in train_loader:

        output = model(input).squeeze()

        loss = loss_fct(output, label)
        loss.backward() 
        optimiser.step()
        optimiser.zero_grad()
        
    if i%freq==0:
        index = int(i/freq)
        for input, label in train_loader:
            with torch.no_grad():
                train_error[index] += loss_fct(model(input).squeeze(),label).item()/num_batch_train
        for input, label in test_loader:
            with torch.no_grad():
                test_error[index] += loss_fct(model(input).squeeze(),label).item()/num_batch_test

print('ok4')
if args.error_save == True:
    torch.save(train_error,'CSV/Spe/mixer/fourth/train_error_MLP_mixer_dim='+str(M)+str(d)+'P='+str(P)+'_sdf='+str(args.seed_function)+'_sds='+str(args.seed_sample)+'_sdm='+str(args.seed_model)+'.pt')
    torch.save(test_error,'CSV/Spe/mixer/fourth/test_error_MLP_mixer_dim='+str(M)+str(d)+'P='+str(P)+'_sdf='+str(args.seed_function)+'_sds='+str(args.seed_sample)+'_sdm='+str(args.seed_model)+'.pt')
    torch.save(param_str,'CSV/Spe/mixer/fourth/param_MLP_mixer_dim='+str(M)+str(d)+'P='+str(P)+'_sdf='+str(args.seed_function)+'_sds='+str(args.seed_sample)+'_sdm='+str(args.seed_model)+'.pt')
if args.model_save == True:
    torch.save(model.state_dict(),'CSV/Spe/mixer/fourth/model_MLP_mixer_dim='+str(M)+str(d)+'P='+str(P)+'_sdf='+str(args.seed_function)+'_sds='+str(args.seed_sample)+'_sdm='+str(args.seed_model)+'.pt')
