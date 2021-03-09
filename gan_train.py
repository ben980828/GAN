import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import imageio
import scipy.misc
import argparse
import glob
import os
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

class Face_Image(Dataset):
    def __init__(self, fileroot, image_root, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.fileroot = fileroot
        self.image_root = image_root
        self.transform = transform

        # read filenames
        self.len = len(self.image_root)       
    def __getitem__(self, index):
        """ Get a sample from the dataset """

        image_fn = self.image_root[index]
        image = Image.open(image_fn).convert('RGB')
        image = self.transform(image) 

        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

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

class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.D_main = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.4),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.4),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.D_main(input)

def main():
    train_root = 'hw3-ben980828/hw3_data/face/train/'
    train_img = []

    train_list = os.listdir(train_root)
    for fn in train_list:
        train_img.append(os.path.join(train_root, fn))
    train_set = Face_Image(fileroot=train_root, 
        image_root=train_img, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5],),
            ])
        )
        
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=1)


    model_G = Generator()
    model_D = Discriminator()
    model_G = model_G.cuda()
    model_D = model_D.cuda()
    model_G.apply(weights_init)
    model_D.apply(weights_init)
    print(model_G)
    print(model_D)

    lr = 2e-4
    #lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max')

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    real_label = random.uniform(0.7, 1.2)
    fake_label = random.uniform(0.0, 0.3)

    optimizerD = optim.SGD(model_D.parameters(), lr=1e-3, momentum=0.9)
    optimizerG = optim.Adam(model_G.parameters(), lr=lr, betas=(0.5, 0.999))

    epoch = 50
    iteration = 0
    G_loss_list = []
    D_loss_list = []
    D_x_list = []
    D_G_list = []
    iter_list = []
    img_list = []
    # training
    for ep in range(1, epoch+1):
        model_D.train()
        model_G.train()
        print('Current training epoch : ', ep)

        for i, data in enumerate(train_loader):
            #################################################################################
            # #1 Training Discriminator
            #################################################################################
            model_D.zero_grad()
            real_data = data.cuda()
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, device=device)
            output = model_D(real_data).view(-1)
            loss_realD = criterion(output, label)
            loss_realD.backward()
            D_x = output.mean().item()
            

            randn_noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_data = model_G(randn_noise)
            label.fill_(fake_label)
            output = model_D(fake_data.detach()).view(-1)
            loss_fakeD = criterion(output, label)
            loss_fakeD.backward()
            D_G_z1 = output.mean().item()

            loss_D = loss_realD + loss_fakeD
            optimizerD.step()
            
            #################################################################################
            # #2 Training Generator
            #################################################################################
            model_G.zero_grad()
            label.fill_(real_label)
            output = model_D(fake_data).view(-1)
            loss_G = criterion(output, label)
            loss_G.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()
            
            # train_loss += loss.item()
            G_loss_list.append(loss_G.item())
            D_loss_list.append(loss_D.item())
            D_x_list.append(D_x)
            D_G_list.append(D_G_z1)
            if (iteration % 500==0):
                with torch.no_grad():
                    fake_output = model_G(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_output, padding=2, normalize=True))
            iteration += 1
            iter_list.append(iteration)
            if (i % 50==0):
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (ep, epoch, i, len(train_loader), loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

        torch.save(model_G.state_dict(), 'gan_G_ep_{}.pth'.format(ep))
        torch.save(model_D.state_dict(), 'gan_D_ep_{}.pth'.format(ep))


    fig, ax = plt.subplots(2, 2, figsize=(30, 20))

    ax[0][0].plot(iter_list, G_loss_list)
    ax[0][0].set_title('Generator Loss')
    ax[0][0].set(xlabel="iteration", ylabel="Loss Value")


    ax[0][1].plot(iter_list, D_loss_list)
    ax[0][1].set_title('Discriminator Loss')
    ax[0][1].set(xlabel="iteration", ylabel="Loss Value")

    ax[1][0].plot(iter_list, D_x_list)
    ax[1][0].set_title('D(x) value')
    ax[1][0].set(xlabel="iteration", ylabel="Value")

    ax[1][1].plot(iter_list, D_G_list)
    ax[1][1].set_title('D(G(z)) value')
    ax[1][1].set(xlabel="iteration", ylabel="Value")


    plt.savefig('Loss_Curve_GAN.png')

    real_batch = next(iter(train_loader))
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig('Real_Fake.png')


if __name__ == '__main__':
    main()

