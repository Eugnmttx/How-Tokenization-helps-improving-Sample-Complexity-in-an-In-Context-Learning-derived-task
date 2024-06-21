
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math as m
from matplotlib import pyplot as plt
from sys import argv
import argparse

import models
import datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

# Parsing for parameters
parser = argparse.ArgumentParser()


parser.add_argument("-m", "--model_type", type=int,
                    help="Choose the type of model, 0 for W, 1 for xy.")
parser.add_argument("-nn", "--NN_type", type=int,
                    help="Choose the architecture of the NN, 0 for linear, 1 for perceptron, 2 for MLP")
parser.add_argument("-b", "--bias", type=bool,
                    help="Set the bias to True or False")
parser.add_argument("-M", "--num_samples", type=int,
                    help="Choose the number of examples/samples in an el. of the train. set")
parser.add_argument("-d", "--dimension", type=int,
                    help="Choose the dimension of the input.")
parser.add_argument("-bs", "--batch_size", type=int,
                    help="Choose the  batch size for SGD.")
parser.add_argument("-lr","--learning_rate", type=float,
                    help="select the learning rate for the training")
parser.add_argument("-T","--num_epochs", type=int,
                    help="Choose the number of epochs for the training")
parser.add_argument("-hl","--hidden_layer",type=int,
                    help="Choose number the number of hidden layer for MLP")
parser.add_argument("-s","--nn_dim",type=int,
                    help="Choose the width of the hidden layers")
parser.add_argument("-f","--frequency",type=int,
                    help="Choose the frequency to store the losses")
parser.add_argument("-sd","--seed_function", type=int,
                    help="Choose the seed for initialising the data function")
parser.add_argument("-sds","--seed_sample", type=int,
                    help="Choose the seed for initialising the data samples")
parser.add_argument("-sdm","--seed_model", type=int,
                    help="Choose the seed for initialising the model")
parser.add_argument("-fl","--file_name", type=str,
                    help="Choose the name of the subfile for saving data")
parser.add_argument("-es","--error_save", type=bool,
                    help="Choose to save the error or not")
parser.add_argument("-ms","--model_save", type=bool,
                    help="Choose to save the model or not")
parser.add_argument("-P","--training_size", type=int,
                    help="Choose the size of the training set")

args = parser.parse_args()

print('ok1')

# Choice of the model
model_type = args.model_type #0 for W model, 1 for xy
NN_type = args.NN_type #0 for Linear, 1 for Perceptron, 2 for MLP

# Parameters for the dataset
Ps = [10, 100, 300, 400, 500, 600, 1000, 5000, 10000, 20000, 25000,  30000, 32768, 33000, 37000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 125000, 150000, 500000] #25
P = Ps[args.training_size]
M = args.num_samples
d = args.dimension
bs = args.batch_size
test_size = P
num_batch_train = m.ceil(P/bs)
num_batch_test = m.ceil(test_size/bs)
flat_len = (M+1)*(d+1)
seed_f = args.seed_function

# Parameters for a MLP
nn_dim = args.nn_dim #dimension of the hidden layers for a MLP

# Dataset
if model_type == 0:
    title = 'W model'
    dataset = datasets.LinearModelW(
        train_size = P,
        num_samples = M,
        dimension = d,
        seed_function=seed_f,
        seed_sample=seed_sample,
        test_size=test_size,
        transform=None
    )
elif model_type == 1:
    title = 'XY model'
    dataset = datasets.LinearModelxy(
        train_size = P,
        num_samples = M,
        dimension = d,
        seed_function=seed_f,
        seed_sample=args.seed_sample,
        test_size=test_size,
        transform=None
    )
else:
    raise ValueError('Model type is not well specified, it should be 1 or 0') from None

print('ok2')

dataset.samples, dataset.labels = dataset.samples.to(device), dataset.labels.to(device)

train_dataset = TensorDataset(dataset.samples[:P].view(P,flat_len), dataset.labels[:P])
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)

test_dataset = TensorDataset(dataset.samples[P:].view(test_size,flat_len), dataset.labels[P:])
test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

# In[]

torch.manual_seed(args.seed_model)

# Model
if NN_type == 0:
    NN_arch = 'Linear NN'
    model = models.MyLinear(
        input_dim=flat_len,
        out_dim = 1
    )
elif NN_type == 1:
    NN_arch = 'Perceptron'
    model = models.Perceptron(
        input_dim = flat_len,
        out_dim = 1,
        norm = 1
    )
elif NN_type == 2:
    NN_arch = 'MLP'
    num_layers = args.hidden_layer #number of layers in the NN for a MLP
    model = models.MLP(
        input_dim=flat_len,
        nn_dim=nn_dim, 
        out_dim=1, 
        num_layers=args.hidden_layer, 
        bias=args.bias, 
        norm= 'mf'
    )
else:
    raise ValueError('NN type is not well specified, it should be 0, 1 or 2') from None

model = model.to(device)
print('ok3')

# Parameters for the training
eta = args.learning_rate
T = int(args.num_epochs)

# Variables for the result analysis
train_error = np.zeros(T)
test_error = np.zeros(T)
freq = args.frequency

# Parameters
param_str = 'M='+str(M)+'d='+str(d)+'P='+str(P)+'width'+str(nn_dim)
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
if args.error_save == True:
    torch.save(train_error,'CSV/Spe/fcn/'+args.file_name+'/train_error_'+title+'_'+NN_arch+'P='+str(P)+'_sdf='+str(seed_f)+'_sds='+str(args.seed_sample)+'_sdm='+str(args.seed_model)+'.pt')
    torch.save(test_error,'CSV/Spe/fcn/'+args.file_name+'/test_error_'+title+'_'+NN_arch+'P='+str(P)+'_sdf='+str(seed_f)+'_sds='+str(args.seed_sample)+'_sdm='+str(args.seed_model)+'.pt')
    torch.save(param_str,'CSV/Spe/fcn/'+args.file_name+'/param_'+title+'_'+NN_arch+'P='+str(P)+'_sdf='+str(seed_f)+'_sds='+str(args.seed_sample)+'_sdm='+str(args.seed_model)+'.pt')
if args.model_save == True:
    torch.save(model.state_dict(),'CSV/Spe/fcn/'+args.file_name+'/model_'+title+'_'+NN_arch+'P='+str(P)+'_sdf='+str(seed_f)+'_sds='+str(args.seed_sample)+'_sdm='+str(args.seed_model)+'.pt')
