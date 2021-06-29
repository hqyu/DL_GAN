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

import numpy as np

# labels = [1, 3, 4, 8, 7, 5, 2, 9, 0, 8, 7]
# # one_hot_index = np.arange(len(labels)) * 10 + labels
# one_hot_index = np.arange(len(labels)) * 10 + labels
# print('one_hot_index:{}'.format(one_hot_index))
#
# one_hot = np.zeros((len(labels), 10))
# print(one_hot.flat[13])
# one_hot.flat[one_hot_index] = 1
#
# print('one_hot:{}'.format(one_hot))
dataroot='./mnist'
DOWNLOAD=False
LR_G=0.0002
LR_D=0.0002
BATCHSIZE=32
LATENT_DIM=100
CLASS_NUM=10
IMG_SIZE=28
EPOCH=50

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
        self.label_embedding = nn.Embedding(10, 10)
        self.layer1=nn.Linear(LATENT_DIM+CLASS_NUM,4*4*128)
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4, 4), 2, 1),
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
    def forward(self,x,labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        x = self.layer1(x)
        x = x.view(-1, 128, 4, 4)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(10, 10)
        self.layer0=nn.Sequential(
            nn.Linear(28*28+10, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 28 * 28),
        )
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
    def forward(self,x,labels):
        c = self.label_embedding(labels)
        # print(c.size())
        x = torch.cat([x, c], 1)
        x = self.layer0(x)
        x=x.view(-1,1,28,28)
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


G_loss=[]
D_loss=[]

iters=0
num=0
for epoch in range(EPOCH):
    for step,inputs in enumerate(dataloader):
        optimizer_D.zero_grad()
        label=torch.full((BATCHSIZE,),1,dtype=torch.float)
        x=inputs[0].view(BATCHSIZE, 784)
        # print(D(x).size())
        labels=inputs[1]
        # labelss=torch.LongTensor([item.detach().numpy() for item in labels])
        # labels.to(torch.int64)
        # print(type(labels))
        output=D(x,labels).view(-1)

        # if step==937:
        #     print(x.size())
        #     print(output.size(),label.size())
        # print(x.size())
        errD_real=loss_func(output,label)
        errD_real.backward()
        D_x=output.mean().item()

        noise = torch.randn(BATCHSIZE, LATENT_DIM)
        condition=torch.randint(0,10,(BATCHSIZE,))
        # conditions=torch.zeros(BATCHSIZE,1,10)
        # # print(conditions.size())
        # for i in range(BATCHSIZE):
        #     conditions[i][0][condition[i]]=1

        # input=torch.cat((noise,condition),1)
        # print(input.size())
        # print(noise.size())
        fake=G(noise,condition).view(BATCHSIZE,784)
        # print(fake.size())
        label.fill_(0)
        output=D(fake.detach(),condition).view(-1)
        errD_fake=loss_func(output,label)
        errD_fake.backward()
        D_G_z1=output.mean().item()
        errD=errD_fake+errD_real
        optimizer_D.step()

        G.zero_grad()
        label.fill_(1)
        output=D(fake,condition).view(-1)
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
                noises = torch.randn(BATCHSIZE, 100)
                fake_labelss = torch.randint(0, 10, (BATCHSIZE,))
                for i in range(BATCHSIZE):
                    fake_labelss[i] = i % 10
                generated_datas = G(noises, fake_labelss)  # batch_size X 784
                save_image(generated_datas.view(generated_datas.size(0), 1, 28, 28),
                           './img2/img_' + (str)(iters) + '.png')

        iters += 1
# for i in range(9):
#     test_z = torch.randn(BATCHSIZE, LATENT_DIM)
#     class_signal = torch.zeros(BATCHSIZE, CLASS_NUM)
#     for j in range(BATCHSIZE):
#         class_signal[j][i] = 1
#     test_z = torch.cat((test_z, class_signal), 1)
#     generated = G(test_z)
#     save_image(generated.view(generated.size(0), 1, 28, 28), './img2/img_test_' + (str)(i) + '.png')
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_loss, label="G")
plt.plot(D_loss, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()