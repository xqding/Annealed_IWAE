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

## read data
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image']
train_label = data['train_label']
test_image = data['test_image']
test_label = data['test_label']

batch_size = 128
train_data = MNIST_Dataset(train_image)

train_data_loader = DataLoader(train_data,
                               batch_size = batch_size,
                               shuffle = True)

test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)

## IWAE models
hidden_size = 10
input_size = train_image.shape[-1]
output_size = train_image.shape[-1]

encoder = IWAE_Encoder(input_size, hidden_size)
decoder = IWAE_Decoder(hidden_size, output_size)

encoder = encoder.cuda()
decoder = decoder.cuda()

## optimizer
encoder_optimizer = optim.Adam(encoder.parameters(), lr = 0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr = 0.001)

## train the model
num_samples = 10
num_epoch = 100
for idx_epoch in range(num_epoch):
    for idx_step, data in enumerate(train_data_loader):
        data = data.cuda()

        ###### train decoder
        x = data.expand(num_samples, data.shape[0], 784)
        mu, sigma = encoder(x)

        #### annealed importance sampling
        epsilon = 0.1
        L = 10

        num_beta = 11
        beta = torch.linspace(0, 1, num_beta)

        ## sample from beta = 0
        h = mu + sigma * torch.randn_like(mu)
        current_q = h.detach().clone()

        _, _, log_QhGx, log_Pxh = calc_energy(current_q, encoder, decoder, x, beta[0])
        log_w = (beta[1] - beta[0])*(log_Pxh - log_QhGx)

        ## sample from beta > 0 and beta < 1
        for k in range(1, num_beta-1):
            flag_accept, current_q = HMC(current_q, encoder, decoder, epsilon, L, x, beta[k])
            _, _, log_QhGx, log_Pxh = calc_energy(current_q, encoder, decoder, x, beta[k])
            log_w += (beta[k+1] - beta[k])*(log_Pxh - log_QhGx)

        ## sample from beta = 1
        flag_accept, current_q = HMC(current_q, encoder, decoder, epsilon, L, x, beta[-1])

        ## calculate annealed importance weights
        log_w = log_w - log_w.max(0)[0]
        w = torch.exp(log_w)
        w = w / w.sum(0)

        ## calculate decoder loss
        h = current_q.data
        log_Pxh = decoder.calc_logPxh(x, h)
        decoder_loss = -torch.mean(torch.sum(w * log_Pxh, 0))

        decoder_optimizer.zero_grad()
        decoder_loss.backward()
        decoder_optimizer.step()

        ###### train encoder
        ## calculate encoder loss
        x = data
        mu, sigma = encoder(x)    
        eps = torch.randn_like(mu)
        h = mu + sigma * eps
        p = decoder(h)
        log_PxGh = torch.sum(x*torch.log(p) + (1-x)*torch.log(1-p), -1)
        DKL = 0.5 * torch.sum(mu**2 + sigma**2 - 1.0 - torch.log(sigma**2), -1)
        encoder_loss = torch.mean(-log_PxGh + DKL)
        encoder_optimizer.zero_grad()
        encoder_loss.backward()
        encoder_optimizer.step()

        print("epoch: {:>3d}, step: {:>5d}, decoder_loss: {:.3f}, encoder_loss: {:.3f}, accept_rate: {:.3f}".format(
            idx_epoch, idx_step, decoder_loss.item(), encoder_loss.item(), flag_accept.float().mean()))

sys.exit()
