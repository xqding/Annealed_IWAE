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

batch_size = 64
train_data = MNIST_Dataset(train_image)

train_data_loader = DataLoader(train_data,
                               batch_size = batch_size,
                               shuffle = True)

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

## optimizer
optimizer = optim.Adam(list(encoder.parameters()) +
                       list(decoder.parameters()),
                       lr = 0.001, eps = 1e-4)
lambda_lr = lambda epoch : 10**(-epoch/7.0)
scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

## train the model
num_samples = args.num_samples

idx_epoch = -1
for i in range(8):
    num_epoch = 3**i
    scheduler_lr.step(i)
    for k in range(num_epoch):
        idx_epoch += 1
        for idx_step, data in enumerate(train_data_loader):
            data = data.cuda()

            ###### calculate IWAE loss
            x = data.expand(num_samples, data.shape[0], 784)
            mu, sigma = encoder(x)
            eps = torch.randn_like(mu)
            h = mu + sigma * eps

            log_Pxh = decoder.calc_logPxh(x, h)
            log_QhGx = torch.sum(-0.5*(eps)**2 -
                                 0.5*torch.log(2*h.new_tensor(np.pi))
                                 - torch.log(sigma), -1)

            log_weight = (log_Pxh - log_QhGx).detach().data
            log_weight = log_weight - torch.max(log_weight, 0)[0]
            weight = torch.exp(log_weight)
            weight = weight / torch.sum(weight, 0)

            loss = -torch.mean(torch.sum(weight*(log_Pxh - log_QhGx), 0))

            ## calculate gradient and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_weight = (log_Pxh - log_QhGx).detach().data
            log_weight = log_weight.double()
            weight = torch.exp(log_weight)
            loss = -torch.mean(torch.log(torch.mean(weight, 0)))
            
            print("epoch: {:>3d}, step: {:>5d}, train_loss: {:.3f}".format(
                idx_epoch, idx_step, loss.item()), flush = True)

        if (idx_epoch + 1) % 100 == 0:
            torch.save({'decoder_state_dict': decoder.state_dict(),
                        'encoder_state_dict': encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       "./output/model/IWAE_hidden_size_{}_num_samples_{}_epoch_{}.pt".format(hidden_size, num_samples, idx_epoch))

torch.save({'decoder_state_dict': decoder.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
           "./output/model/IWAE_hidden_size_{}_num_samples_{}_epoch_{}.pt".format(hidden_size, num_samples, idx_epoch))

sys.exit()
