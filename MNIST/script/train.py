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

batch_size = 256
train_data = MNIST_Dataset(train_image)

train_data_loader = DataLoader(train_data,
                               batch_size = batch_size,
                               shuffle = True)

test_data = MNIST_Dataset(test_image)
test_data_loader = DataLoader(test_data, batch_size = batch_size)


hidden_size = 10
input_size = train_image.shape[-1]
output_size = train_image.shape[-1]

encoder = IWAE_Encoder(input_size, hidden_size)
decoder = IWAE_Decoder(hidden_size, output_size)

encoder_optimizer = optim.Adam(encoder.parameters(), lr = 0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr = 0.001)

num_samples = 8
for idx, data in enumerate(train_data_loader):
    x = data.expand(num_samples, batch_size, 784)

    ## encode
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
    decoder_loss = torch.mean(torch.sum(w * log_Pxh, 0))
    
    decoder_optimizer.zero_grad()    
    decoder_loss.backward()
    decoder_optimizer.step()

    break


sys.exit()

if args.num_stochastic_layers == 1:
    vae = IWAE_1(50, 784)
elif args.num_stochastic_layers == 2:
    vae = IWAE_2(100, 50, 784)
    
vae.double()
vae.cuda()

optimizer = optim.Adam(vae.parameters())
num_epoches = 5000
train_loss_epoch = []
for epoch in range(num_epoches):
    running_loss = []    
    for idx, data in enumerate(train_data_loader):
        data = data.double()
        inputs = Variable(data).cuda()
        if args.model == "IWAE":
            inputs = inputs.expand(args.num_samples, batch_size, 784)
        elif args.model == "VAE":
            inputs = inputs.repeat(args.num_samples, 1)
            inputs = inputs.expand(1, batch_size*args.num_samples, 784)
            
        optimizer.zero_grad()
        loss = vae.train_loss(inputs)
        loss.backward()
        optimizer.step()    
        print(("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}")
              .format(epoch, idx, loss.item()), flush = True)
        running_loss.append(loss.item())


    train_loss_epoch.append(np.mean(running_loss))

    if (epoch + 1) % 1000 == 0:
        torch.save(vae.state_dict(),
                   ("./output/model/{}_layers_{}_k_{}_epoch_{}.model")
                   .format(args.model, args.num_stochastic_layers,
                           args.num_samples, epoch))

