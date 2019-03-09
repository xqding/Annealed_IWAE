import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
sys.path.append("../model/")
from vae_models import *
from functions import *
from sys import exit
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec

parser = argparse.ArgumentParser(description="Annealed Importance Weighted Auto-Encoder")
parser.add_argument("--hidden_size", type = int,
                    required = True)
parser.add_argument("--num_samples", type = int,
                    required = True,
                    help = """num of samples used in Monte Carlo estimate of 
                              ELBO when using VAE; num of samples used in 
                              importance weighted ELBO when using IWAE.""")
args = parser.parse_args()

print("hidden_size: {}, num_samples: {}".format(args.hidden_size, args.num_samples))

####  read data
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image']
train_label = data['train_label']
test_image = data['test_image']
test_label = data['test_label']

batch_size = 1024
train_data = MNIST_Dataset(train_image)
train_data_loader = DataLoader(train_data, batch_size = batch_size)

## IWAE models
hidden_size = args.hidden_size
input_size = train_image.shape[-1]
output_size = train_image.shape[-1]

encoder = IWAE_Encoder(input_size, hidden_size)
decoder = IWAE_Decoder(hidden_size, output_size)

encoder = encoder.cuda()
decoder = decoder.cuda()

num_samples = args.num_samples

## IWAE state dict
state_dict = torch.load("./output/model/IWAE_hidden_size_{}_num_samples_{}_epoch_3279.pt".format(hidden_size, num_samples))
encoder.load_state_dict(state_dict['encoder_state_dict'])
decoder.load_state_dict(state_dict['decoder_state_dict'])

h_mu = []
for idx_step, data in enumerate(train_data_loader):
    print(idx_step)
    data = data.cuda()
    mu, sigma = encoder(data)
    mu = mu.detach().cpu().data.numpy()
    h_mu.append(mu)

h_mu = np.vstack(h_mu)
h_mu_var_IWAE = np.std(h_mu, 0)**2
print("IWAE num of active units (var >= 0.01): {}".format(np.sum(h_mu_var_IWAE >= 0.01)))

## AIWAE state dict
state_dict = torch.load("./output/model/AIWAE_hidden_size_{}_num_samples_{}_epoch_599.pt".format(hidden_size, num_samples))
encoder.load_state_dict(state_dict['encoder_state_dict'])
decoder.load_state_dict(state_dict['decoder_state_dict'])

h_mu = []
for idx_step, data in enumerate(train_data_loader):
    print(idx_step)
    data = data.cuda()
    mu, sigma = encoder(data)
    mu = mu.detach().cpu().data.numpy()
    h_mu.append(mu)

h_mu = np.vstack(h_mu)
h_mu_var_AIWAE = np.std(h_mu, 0)**2
print("AIWAE num of active units (var >= 0.01): {}".format(np.sum(h_mu_var_AIWAE >= 0.01)))

fig = plt.figure(0)
fig.clf()
plt.hist(np.log(h_mu_var_IWAE), alpha = 0.5, label = 'IWAE', bins = 10)
plt.hist(np.log(h_mu_var_AIWAE), alpha = 0.5, label = 'AIWAE', bins = 10)
plt.xlabel('log of variance of h_mu')
plt.legend()
plt.savefig('./output/num_active_unit.pdf')

# state_dict = torch.load("./output/model/IWAE_hidden_size_{}_num_samples_{}_epoch_3279.pt".format(hidden_size, num_samples))
# encoder.load_state_dict(state_dict['encoder_state_dict'])
# decoder.load_state_dict(state_dict['decoder_state_dict'])
