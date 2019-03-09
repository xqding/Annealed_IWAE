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

parser = argparse.ArgumentParser(description="Importance Weighted Auto-Encoder")
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

batch_size = 2048
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
state_dict = torch.load("./output/model/IWAE_hidden_size_{}_num_samples_{}_epoch_3279.pt".format(hidden_size, num_samples))
encoder.load_state_dict(state_dict['encoder_state_dict'])
decoder.load_state_dict(state_dict['decoder_state_dict'])

loss_all = []
num_beta = 1000
betas = np.linspace(0, 1, num_beta)
L = 10
epsilon = 0.1
epsilon_min = 0.01
epsilon_max = 0.3
epsilon_increase_alpha = 1.02
epsilon_decrease_alpha = 0.98
epsilon_target = 0.65

NLL = []

for idx_step, data in enumerate(test_data_loader):
    print(idx_step)    
    data = data.cuda()

    ###### calculate IWAE loss
    x = data.expand(16, data.shape[0], 784)
    mu, sigma = encoder(x)
    eps = torch.randn_like(mu)
    h = mu + sigma * eps
    
    log_w = 0
    epsilon = 0.1
    for i in range(num_beta-1):
        log_Pxh = decoder.calc_logPxh(x, h)
        log_QhGx = encoder.calc_logQhGx(x, h)
        
        log_w += (betas[i+1] - betas[i])*(log_Pxh.data - log_QhGx.data)

        flag_accept, h = HMC(h.detach().clone(), encoder, decoder, epsilon, L, x, betas[i+1])

        accept_rate = flag_accept.float().mean().item()
        if accept_rate > epsilon_target:
            epsilon *= epsilon_increase_alpha
        else:
            epsilon *= epsilon_decrease_alpha
            
        print("beta_idx: {:>4d}, beta: {:.2f}, accept_rate: {:.3f}, epsilon: {:.3f}".format(i, betas[i+1], accept_rate, epsilon))

    log_w = log_w.double()
    log_w_min = log_w.min(0)[0]
    log_w = log_w - log_w_min
    w = torch.exp(log_w)
    nll = -(torch.log(torch.mean(w, 0)) + log_w_min)
    nll = list(nll.detach().cpu().data.numpy())
    NLL += nll

sys.exit()
