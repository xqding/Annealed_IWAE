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

num_gene_samples = 100
idx = np.random.choice(range(train_image.shape[0]), size = num_gene_samples, replace = False)
data = train_image[idx]

fig = plt.figure(0, figsize = (10,10))
fig.clf()
gs = gridspec.GridSpec(int(np.sqrt(num_gene_samples)), int(np.sqrt(num_gene_samples)))
gs.update(wspace = 0.05, hspace = 0.05)
for i in range(num_gene_samples):
    axes = plt.subplot(gs[i])
    axes.imshow(1-data[i].reshape(28,28),
                cmap = 'binary', vmin = 0, vmax = 1)
    axes.axis('off')
plt.savefig("./output/sample/train_sample.pdf")    

## IWAE models
hidden_size = args.hidden_size
input_size = train_image.shape[-1]
output_size = train_image.shape[-1]

encoder = IWAE_Encoder(input_size, hidden_size)
decoder = IWAE_Decoder(hidden_size, output_size)

encoder = encoder.cuda()
decoder = decoder.cuda()

num_samples = args.num_samples
state_dict = torch.load("./output/model/AIWAE_hidden_size_{}_num_samples_{}_epoch_599.pt".format(hidden_size, num_samples))
encoder.load_state_dict(state_dict['encoder_state_dict'])
decoder.load_state_dict(state_dict['decoder_state_dict'])

h = torch.randn(num_gene_samples, hidden_size)
h = h.cuda()
p = decoder(h)
p = p.detach().cpu().data.numpy()
#p = np.random.rand(num_gene_samples, output_size) < p

fig = plt.figure(0, figsize = (10,10))
fig.clf()
gs = gridspec.GridSpec(int(np.sqrt(num_gene_samples)), int(np.sqrt(num_gene_samples)))
gs.update(wspace = 0.05, hspace = 0.05)
for i in range(num_gene_samples):
    axes = plt.subplot(gs[i])
    axes.imshow(1-p[i].reshape(28,28),
                cmap = 'binary', vmin = 0, vmax = 1)
    axes.axis('off')
plt.savefig("./output/sample/AIWAE_sample.pdf")    

state_dict = torch.load("./output/model/IWAE_hidden_size_{}_num_samples_{}_epoch_3279.pt".format(hidden_size, num_samples))
encoder.load_state_dict(state_dict['encoder_state_dict'])
decoder.load_state_dict(state_dict['decoder_state_dict'])

# h = torch.randn(num_samples, hidden_size)
# h = h.cuda()
p = decoder(h)
p = p.detach().cpu().data.numpy()
#p = np.random.rand(num_samples, output_size) < p

fig = plt.figure(0, figsize = (10,10))
fig.clf()
gs = gridspec.GridSpec(int(np.sqrt(num_gene_samples)), int(np.sqrt(num_gene_samples)))
gs.update(wspace = 0.05, hspace = 0.05)
for i in range(num_gene_samples):
    axes = plt.subplot(gs[i])
    axes.imshow(1-p[i].reshape(28,28),
                cmap = 'binary', vmin = 0, vmax = 1)
    axes.axis('off')
plt.savefig("./output/sample/IWAE_sample.pdf")    

#plt.show()   
sys.exit()

