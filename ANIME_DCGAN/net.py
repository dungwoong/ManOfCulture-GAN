import torch
import torch.nn as nn
from config import cfg

nz = cfg["Z_dim"]
ngpu = cfg["num_gpus"]
ngf = cfg["generator_depth"]
nc = cfg["img_channels"]
ndf = cfg["discriminator_depth"]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = ngpu  # number of GPUs available
        # since he feeds the input in as a convolution, he can put everything in a sequential model
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=nz,
                               out_channels=ngf * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0, bias=False),
            # ngf is number of feature maps in the generator.
            nn.BatchNorm2d(num_features=ngf * 8),
            # I think if you think of feature map as random matrix, it will keep track of mean/var for each pixel in
            # the map
            nn.ReLU(True),  # inplace relu
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
            # output channels is 3
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=nc,
                      out_channels=ndf,
                      kernel_size=4,
                      stride=2,
                      padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # outputs 1 feature map that is 1x1
            nn.Sigmoid()
        )

    # convolutional classifier, pretty simple

    def forward(self, x):
        return self.main(x)
