import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision.utils import save_image

EPOCH=20
LR_G=0.0002
LR_D=0.0002
IMG_SIZE=28
LATENT_DIM=100
BATCHSIZE=32
DATA_ROOT='./mnist'
DOWNLOAD=False

dataloader=data.DataLoader(
    datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        transform=transforms.ToTensor(),
        download=DOWNLOAD
    ),
    batch_size=BATCHSIZE,
    shuffle=True
)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(LATENT_DIM,128),
            nn.LeakyReLU(0.2)
        )
        self.layer2=nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(512, IMG_SIZE*IMG_SIZE),
            nn.Tanh()
        )
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),1,IMG_SIZE,IMG_SIZE)
        return x
class Discrimator(nn.Module):
    def __init__(self):
        super(Discrimator, self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(IMG_SIZE*IMG_SIZE,512),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=x.view(-1,IMG_SIZE*IMG_SIZE)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        return x
G=Generator()
D=Discrimator()

real_label=1
fake_label=0

optimizer_G=optim.Adam(G.parameters(),lr=LR_G)
optimizer_D=optim.Adam(D.parameters(),lr=LR_D)

loss_func=nn.BCELoss()

loss_G=[]
loss_D=[]
iters=0
for epoch in range(EPOCH):
    for step,(x,y)in enumerate(dataloader):
        optimizer_D.zero_grad()
        label=torch.full((BATCHSIZE,),real_label,dtype=torch.float)
        # print(x.size())
        output=D(x).view(-1)
        # print(output.size())
        # print(label.size())
        loss_real=loss_func(output,label)
        loss_real.backward()

        noise=torch.randn(BATCHSIZE,LATENT_DIM)
        label=torch.full((BATCHSIZE,),fake_label,dtype=torch.float)
        input=G(noise)
        output=D(input.detach()).view(-1)
        loss_fake=loss_func(output,label)
        loss_fake.backward()
        loss_d=loss_real+loss_fake
        optimizer_D.step()

        optimizer_G.zero_grad()
        label.fill_(real_label)
        output=D(input).view(-1)
        loss_g=loss_func(output,label)
        loss_g.backward()
        optimizer_G.step()

        if step % 300 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch + 1, EPOCH, step, len(dataloader),
                     loss_d.item(), loss_g.item()))

            # Save Losses for plotting later
        loss_G.append(loss_g.item())
        loss_D.append(loss_d.item())
        if (iters % 500 == 0) or ((epoch == EPOCH - 1) and (step == len(dataloader) - 1)):
            with torch.no_grad():
                test_z = torch.randn(BATCHSIZE, LATENT_DIM)
                generated = G(test_z)
                save_image(generated.view(generated.size(0), 1, 28, 28), './img4/img_' + (str)(iters) + '.png')

        iters += 1

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(loss_G, label="G")
plt.plot(loss_D, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


