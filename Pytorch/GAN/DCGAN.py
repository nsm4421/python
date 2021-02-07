#---- Load modules -----#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision import datasets
import torchvision.transforms as tr
from torchvision.utils import save_image

import matplotlib.pyplot as plt

#---- Aruments ----#

LR = 2e-4
N_EPOCH = 200
LEN_Z = 100
BATCH_SIZE = 128

N_CHANNEL = 1
IMG_SIZE = 28

#---- Datasets ----#

# Transform
transforms = tr.Compose([tr.Resize(IMG_SIZE),
                               tr.ToTensor(),
                               tr.Normalize([0.5], [0.5])])
# Download dataset
MNIST = datasets.MNIST(root="./dataset",
                       train=True,
                       download=True,
                       transform=transforms)

# Data Loader
dataloader = DataLoader(MNIST, batch_size = BATCH_SIZE, shuffle=True)

#---- Look up dataset ----#

# sample = next(iter(MNIST))
# plt.imshow(sample[0].squeeze(0), cmap = 'gray')
# plt.title(f'label : {sample[1]} / shape : {sample[0].squeeze(0).shape}')
# plt.show()

#---- Generator ----#

class Generator(nn.Module):
  def __init__(self, n_channel = N_CHANNEL, img_size = IMG_SIZE, len_z = LEN_Z):
    super(Generator, self).__init__()
    self.n_channel = N_CHANNEL
    self.img_size = IMG_SIZE
    self.len_z = LEN_Z

    def block(input_dim, output_dim,
              normalize=True,momentum = 0.8, slope = 0.2):
      layers = [nn.Linear(input_dim, output_dim)]
      if normalize:
          layers.append(nn.BatchNorm1d(output_dim, momentum))
      layers.append(nn.LeakyReLU(slope, inplace=True))
      return layers

    self.generator = nn.Sequential(
        *block(self.len_z, 128, normalize=False),
        *block(128, 256),
        *block(256, 512),
        *block(512, 1024),
        nn.Linear(1024, self.n_channel*self.img_size**2),
        nn.Tanh(),)

  def forward(self, z):
      fake_img = self.generator(z)
      fake_img = fake_img.view(fake_img.size(0), self.n_channel, self.img_size, self.img_size)
      return fake_img

#---- Discriminator ----#

class Discriminator(nn.Module):
  def __init__(self, n_channel = N_CHANNEL, img_size = IMG_SIZE):
    super(Discriminator, self).__init__()

    self.n_channel = n_channel
    self.img_size = img_size

    self.discriminator = nn.Sequential(nn.Linear(self.n_channel * self.img_size**2, 512),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Linear(512, 256),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Linear(256, 1),
                                       nn.Sigmoid(),)

  def forward(self, img):
      flattened = img.view(img.size(0), -1)
      output = self.discriminator(flattened)

      return output

G = Generator()
D = Discriminator()


#---- Training function----#

class train:

  def __init__(self, generator = G, discriminator = D,
               len_z = LEN_Z, lr = LR, betas = (0.5, 0.99)):
    
    # initialize
    self.G = G    # generator
    self.D = D    # discriminator
    self.len_z = len_z    # length of latent vector
    self.lr = lr    # learning rate
    self.betas = betas    # argument for adam optimizer
    
    # loss & optimizer
    self.loss = nn.BCELoss()    # binary cross entropy
    self.optimizer_G = Adam(self.G.parameters(), lr = self.lr, betas = self.betas)
    self.optimizer_D = Adam(self.D.parameters(), lr = self.lr, betas = self.betas)

  def training_loop(self, dataloader = dataloader, n_epoch = 200, save = False):
  
    for epoch in range(n_epoch):
      for batch in dataloader:

        # unpack
        real_images, _ = batch
        batch_size = real_images.size(0)
        save_term = batch_size // 10

        # label
        real_labels = torch.FloatTensor(batch_size, 1).fill_(1.0)
        fake_labels = torch.FloatTensor(batch_size, 1).fill_(0.0)

        # initailize
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        ##---- Training Generator ----##

        z = torch.normal(mean=0, std=1, size=(batch_size, self.len_z))
        
        # generate fake image
        fake_images = self.G(z)
        fake_outputs = self.D(fake_images)

        # caculate generator loss
        G_loss = self.loss(fake_outputs, real_labels)
        
        # update generator
        G_loss.backward()
        self.optimizer_G.step()

        ##---- Training Discriminator ----##
        
        # caculate discriminator loss
        D_real_loss = self.loss(self.D(real_images), real_labels)
        D_fake_loss = self.loss(self.D(fake_images.detach()), fake_labels)
        D_loss = D_real_loss + D_fake_loss
        
        # update discriminator
        D_loss.backward()
        self.optimizer_D.step()

      # print loss
      print(f' epoch : {epoch} / Generator loss {G_loss.item():.6f} / Discriminator loss: {D_loss.item():.6f}')

      # save fake images
      if save and epoch%save_term == 0:
        save_image(fake_images.data[:16], f"{epoch}.png", nrow=4, normalize=True)


# Run training loop
train_function = train()
train_function.training_loop(n_epoch = N_EPOCH, save = True)

# model
G = train_function.G
D = train_function.D

