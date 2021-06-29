import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision.utils import save_image

dataroot='./mnist'
DOWNLOAD=False
LR_G=0.0002
LR_D=0.0002
BATCHSIZE=32
LATENT_DIM=100
IMG_SIZE=28
EPOCH=10

train_data=datasets.MNIST(
    root=dataroot,
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD
)
dataloader=torch.utils.data.DataLoader(
    train_data,batch_size=BATCHSIZE,shuffle=True
)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(LATENT_DIM,4*4*128)
        )
        self.layer2=nn.Sequential(
            nn.ConvTranspose2d(128,64,(4,4),2,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4, 4), 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (4, 4), 2, 3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, (1, 1), 1, 0),
            nn.Tanh()
        )
    def forward(self,x):
        x=self.layer1(x)
        x=x.view(-1,128,4,4)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,32,(4,4),2,3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, (4, 4), 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, (4, 4), 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 1, (4, 4), 1, 0),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        return x

G=Generator()
D=Discriminator()

optimizer_G=torch.optim.Adam(G.parameters(),lr=LR_G)
optimizer_D=torch.optim.Adam(D.parameters(),lr=LR_D)

loss_func=nn.BCELoss()


real_label=1
fake_label=0

img_list=[]
G_loss=[]
D_loss=[]
iters=0
for epoch in range(EPOCH):
    for step,(x,_)in enumerate(dataloader):
        optimizer_D.zero_grad()
        label=torch.full((BATCHSIZE,),real_label,dtype=torch.float)
        output=D(x).view(-1)
        # if step==937:
        #     print(x.size())
        #     print(output.size(),label.size())
        # print(x.size())
        errD_real=loss_func(output,label)
        errD_real.backward()
        D_x=output.mean().item()

        noise = torch.randn(BATCHSIZE, LATENT_DIM)
        # print(noise.size())
        fake=G(noise)
        # print(fake.size())
        label.fill_(fake_label)
        output=D(fake.detach()).view(-1)
        errD_fake=loss_func(output,label)
        errD_fake.backward()
        D_G_z1=output.mean().item()
        errD=errD_fake+errD_real
        optimizer_D.step()

        G.zero_grad()
        label.fill_(real_label)
        output=D(fake).view(-1)
        errG=loss_func(output,label)
        errG.backward()
        D_G_z2=output.mean().item()
        optimizer_G.step()
        if step % 300 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch+1, EPOCH, step, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
        G_loss.append(errG.item())
        D_loss.append(errD.item())
        if (iters % 500 == 0) or ((epoch == EPOCH - 1) and (step == len(dataloader) - 1)):
            with torch.no_grad():
                test_z=torch.randn(BATCHSIZE,LATENT_DIM)
                generated=G(test_z)
                save_image(generated.view(generated.size(0),1,28,28),'./img/img_'+(str)(iters)+'.png')

        iters += 1

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_loss, label="G")
plt.plot(D_loss, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()