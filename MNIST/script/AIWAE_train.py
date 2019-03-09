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

## IWAE models
hidden_size = args.hidden_size
input_size = train_image.shape[-1]
output_size = train_image.shape[-1]

encoder = IWAE_Encoder(input_size, hidden_size)
decoder = IWAE_Decoder(hidden_size, output_size)

encoder = encoder.cuda()
decoder = decoder.cuda()

## optimizer
encoder_optimizer = optim.Adam(encoder.parameters(),
                               lr = 0.001,
                               eps = 1e-4)
decoder_optimizer = optim.Adam(decoder.parameters(),
                               lr = 0.001,
                               eps = 1e-4)
lambda_lr = lambda epoch : 10**(-epoch/7.0)

encoder_scheduler_lr = optim.lr_scheduler.LambdaLR(encoder_optimizer, lambda_lr)
decoder_scheduler_lr = optim.lr_scheduler.LambdaLR(decoder_optimizer, lambda_lr)

## train the model
num_samples = args.num_samples

## parameters for HMC
epsilon = 0.1
epsilon_min = 0.01
epsilon_max = 0.3
epsilon_decrease_alpha = 0.998
epsilon_increase_alpha = 1.002
epsilon_target = 0.6

L = 5
num_beta = 10
beta = torch.linspace(0, 1, num_beta)

idx_epoch = -1
for i in range(8):
    num_epoch = 3**i
    encoder_scheduler_lr.step(i)
    decoder_scheduler_lr.step(i)

    for k in range(num_epoch):
        idx_epoch += 1
        
        for idx_step, data in enumerate(train_data_loader):
            data = data.cuda()

            ###### train decoder
            x = data.expand(num_samples, data.shape[0], 784)
            mu, sigma = encoder(x)

            #### annealed importance sampling
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

            if flag_accept.float().mean().item() < epsilon_target:
                epsilon *= epsilon_decrease_alpha
            else:
                epsilon *= epsilon_increase_alpha

            if epsilon < epsilon_min:
                epsilon = epsilon_min
            if epsilon > epsilon_max:
                epsilon = epsilon_max
                
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

            ## calculate importance weighted loss
            x = data.expand(num_samples, data.shape[0], 784)
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
            importance_loss = -torch.mean(torch.log(torch.mean(weight, 0)))                        

            print("epoch: {:>3d}, step: {:>5d}, decoder_loss: {:.3f}, encoder_loss: {:.3f}, importance_loss: {:.3f}, accept_rate: {:.3f}, epsilon: {:.3f}".format(
                idx_epoch, idx_step, decoder_loss.item(), encoder_loss.item(), importance_loss.item(), flag_accept.float().mean(), epsilon), flush = True)

        if (idx_epoch + 1) % 20 == 0:
            torch.save({'decoder_state_dict': decoder.state_dict(),
                        'encoder_state_dict': encoder.state_dict(),
                        'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                        'encoder_optimizer_state_dict': encoder_optimizer.state_dict()},
                       "./output/model/AIWAE_hidden_size_{}_num_samples_{}_epoch_{}.pt".format(hidden_size, num_samples, idx_epoch))

torch.save({'decoder_state_dict': decoder.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict()},
           "./output/model/AIWAE_hidden_size_{}_num_samples_{}_epoch_{}.pt".format(hidden_size, num_samples, idx_epoch))
            
            
        
sys.exit()
