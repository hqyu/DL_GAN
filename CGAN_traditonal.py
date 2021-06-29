
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch import optim as optim
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_parameters = {
    "n_epochs": 100,
    "batch_size": 100,
}
data_loader = torch.utils.data.DataLoader(

    datasets.MNIST('./mnist', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(
                           (0.5,), (0.5,))
                   ])),
    batch_size=training_parameters["batch_size"], shuffle=True)
num_batches = len(data_loader)
print("Number of batches: ",num_batches)


class GeneratorModel(nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()
        input_dim = 100 + 10
        output_dim = 784
        self.label_embedding = nn.Embedding(10, 10)

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2)
        )

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        # print(c.size())
        x = torch.cat([x, c], 1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)
        return output.to(device)
class DiscriminatorModel(nn.Module):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        input_dim = 784 + 10
        output_dim = 1
        self.label_embedding = nn.Embedding(10, 10)

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)

        return output.to(device)

discriminator = DiscriminatorModel()
generator = GeneratorModel()
discriminator.to(device)
generator.to(device)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()
n_epochs = training_parameters["n_epochs"]
batch_size = training_parameters["batch_size"]
iters=0
for epoch_idx in range(n_epochs):
    G_loss = []
    D_loss = []
    for batch_idx, data_input in enumerate(data_loader):
        noise = torch.randn(batch_size, 100).to(device)
        fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
        # print(fake_labels)
        generated_data = generator(noise, fake_labels)  # batch_size X 784

        # Discriminator
        true_data = data_input[0].view(batch_size, 784).to(device)  # batch_size X 784
        digit_labels = data_input[1].to(device)  # batch_size
        true_labels = torch.ones(batch_size).to(device)

        discriminator_optimizer.zero_grad()

        discriminator_output_for_true_data = discriminator(true_data, digit_labels).view(batch_size)
        true_discriminator_loss = loss(discriminator_output_for_true_data, true_labels)

        discriminator_output_for_generated_data = discriminator(generated_data.detach(), fake_labels).view(batch_size)
        generator_discriminator_loss = loss(
            discriminator_output_for_generated_data, torch.zeros(batch_size).to(device)
        )
        discriminator_loss = (
                                     true_discriminator_loss + generator_discriminator_loss
                             ) / 2

        discriminator_loss.backward()
        discriminator_optimizer.step()

        D_loss.append(discriminator_loss.data.item())

        # Generator

        generator_optimizer.zero_grad()
        # It's a choice to generate the data again
        generated_data = generator(noise, fake_labels)  # batch_size X 784
        discriminator_output_on_generated_data = discriminator(generated_data, fake_labels).view(batch_size)
        generator_loss = loss(discriminator_output_on_generated_data, true_labels)
        generator_loss.backward()
        generator_optimizer.step()
        G_loss.append(generator_loss.data.item())
        # if batch_idx % 300 == 0:
        #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
        #           % (epoch_idx+1, n_epochs, batch_idx, len(data_loader),
        #              discriminator_loss, generator_loss.item()))
        if (iters % 500 == 0) or ((epoch_idx == n_epochs - 1) and (batch_idx== len(data_loader) - 1)):
            with torch.no_grad():
                noises = torch.randn(batch_size, 100).to(device)
                fake_labelss = torch.randint(0, 10, (batch_size,)).to(device)
                for i in range(batch_size):
                    fake_labelss[i]=i%10
                generated_datas = generator(noises, fake_labelss)  # batch_size X 784
                save_image(generated_datas.view(generated_datas.size(0),1,28,28),'./img3/img_'+(str)(iters)+'.png')

        iters += 1
        if ((batch_idx + 1) % 500 == 0 and (epoch_idx + 1) % 10 == 0):
            print("Training Steps Completed: ", batch_idx)

            with torch.no_grad():
                noise = torch.randn(batch_size, 100).to(device)
                fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
                generated_data = generator(noise, fake_labels).cpu().view(batch_size, 28, 28)
                for x in generated_data:
                    print(fake_labels[0].item())
                    plt.imshow(x.detach().numpy(), interpolation='nearest', cmap='gray')
                    plt.show()

                    break
    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch_idx), n_epochs, torch.mean(torch.FloatTensor(D_loss)), torch.mean(torch.FloatTensor(G_loss))))