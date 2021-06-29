import argparse
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
N_CRITIC=5

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
# parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
# opt = parser.parse_args()
# print(opt)

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

optimizer_G=optim.RMSprop(G.parameters(),lr=LR_G)
optimizer_D=optim.RMSprop(D.parameters(),lr=LR_D)



loss_G=[]
loss_D=[]
iters=0
for epoch in range(EPOCH):
    for step,(x,y)in enumerate(dataloader):

        # print(0,D.parameters())
        for p in D.parameters():
            p.data.clamp_(-0.01,0.01)
            # print(p)
        optimizer_D.zero_grad()
        # label = torch.full((BATCHSIZE,), real_label, dtype=torch.float)
            # print(x.size())
        real = D(x).reshape(-1)
            # print(output.size())
            # print(label.size())

        noise = torch.randn(BATCHSIZE, LATENT_DIM)
            # label=torch.full((BATCHSIZE,),fake_label,dtype=torch.float)
        fake = G(noise)
        output = D(fake.detach()).reshape(-1)

        loss_d =-(torch.mean(real) - torch.mean(output))
        loss_d.backward()
        optimizer_D.step()
            # Train the generator every n_critic iterations



        optimizer_G.zero_grad()
            # label.fill_(real_label)
        output=D(fake).reshape(-1)
        loss_g=-torch.mean(output)
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
                save_image(generated.view(generated.size(0), 1, 28, 28), './img5/img_' + (str)(iters) + '.png')

        iters += 1

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(loss_G, label="G")
plt.plot(loss_D, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


