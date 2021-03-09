import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import imageio
import scipy.misc
import argparse
import glob
import os
import sys
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import torchvision.utils as vutils

torch.manual_seed(0)


# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)


class Generator(nn.Module):
    def __init__(self, in_channel=100):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            nn.Dropout2d(0.4), 
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.Dropout2d(0.4), 
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            nn.Dropout2d(0.4), 
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True), 
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)



def making_latent(mu, var):
    std = var
    eps = torch.randn_like(std)
    return (mu + eps*std)

def main():
    # pyfile = sys.argv[0]
    # output_file = sys.argv[1]
    output_file = 'GAN_generate_img.png'
    generator = Generator()
    generator = generator.cuda()

    print(generator)
    generate_img = []
    state = torch.load('gan_G_optimal_ep_40.pth')
    generator.load_state_dict(state)
    with torch.no_grad():
        mu = torch.FloatTensor(32, 100).fill_(0.0).cuda()
        var = torch.FloatTensor(32, 100).fill_(1.0).cuda()
        z = making_latent(mu, var).cuda()
        generator.eval()
        z = torch.unsqueeze(z, 2)
        input_vec = torch.unsqueeze(z, 3)
        output_img = generator(input_vec)
        output_img = output_img.cpu().detach()
        generate_img.append(vutils.make_grid(output_img, padding=2, normalize=True))

        plt.figure(figsize=(15,15))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(generate_img[-1],(1,2,0)))
        plt.savefig(output_file)


if __name__ == '__main__':
    main()

