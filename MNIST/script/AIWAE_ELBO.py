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

batch_size = 16
test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)

## IWAE models
hidden_size = args.hidden_size
input_size = train_image.shape[-1]
output_size = train_image.shape[-1]

encoder = IWAE_Encoder(input_size, hidden_size)
decoder = IWAE_Decoder(hidden_size, output_size)

encoder = encoder.cuda()
decoder = decoder.cuda()

num_samples = args.num_samples
state_dict = torch.load("./output/model/AIWAE_hidden_size_{}_num_samples_{}_epoch_479.pt".format(hidden_size, num_samples))
encoder.load_state_dict(state_dict['encoder_state_dict'])
decoder.load_state_dict(state_dict['decoder_state_dict'])

loss_all = []
for idx_step, data in enumerate(test_data_loader):
    data = data.cuda()

    ###### calculate IWAE loss
    x = data.expand(5000, data.shape[0], 784)
    mu, sigma = encoder(x)
    eps = torch.randn_like(mu)
    h = mu + sigma * eps

    log_Pxh = decoder.calc_logPxh(x, h)
    log_QhGx = torch.sum(-0.5*(eps)**2 -
                         0.5*torch.log(2*h.new_tensor(np.pi))
                         - torch.log(sigma), -1)

    log_weight = (log_Pxh - log_QhGx).detach().data
    log_weight = log_weight.double()
    weight = torch.exp(log_weight)
    loss = -torch.log(torch.mean(weight, 0))

    loss_all += list(loss.detach().cpu().data.numpy())
    print(idx_step)

print("Mean ELBO: {:.2f}".format(np.mean(loss_all)))

sys.exit()
