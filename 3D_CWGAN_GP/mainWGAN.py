import random
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from WGAN3D_model import Generator, Discriminator
from training3DWGAN import Trainer
from Training_Image_3D import training_image_prepare
import cv2

train_dataloader, test_dataloader = training_image_prepare()

generator = Generator()
discriminator = Discriminator(dim=32)

G_optimizer = optim.Adam(generator.parameters(), lr=0.00002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))

epochs = 160
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, use_cuda=torch.cuda.is_available())
trainer.train(train_dataloader, epochs)


trainer.plot_loss(trainer.losses['total_G'], trainer.losses['D'])
