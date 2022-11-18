import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


def initialize_weights(m):
    if isinstance(m, nn.Conv3d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(256)
        self.conv5 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn5 = nn.BatchNorm3d(512)
        self.conv6 = nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

        # 定义反卷积层
        self.trans_conv1 = nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=(4, 4, 4), stride=(2, 2, 2),
                                              padding=(1, 1, 1))
        self.bn6 = nn.BatchNorm3d(512)

        self.trans_conv2 = nn.ConvTranspose3d(in_channels=1024, out_channels=256, kernel_size=(4, 4, 4), stride=(2, 2, 2),
                                              padding=(1, 1, 1))
        self.bn7 = nn.BatchNorm3d(256)
        self.trans_conv3 = nn.ConvTranspose3d(in_channels=512, out_channels=128, kernel_size=(4, 4, 4), stride=(2, 2, 2),
                                              padding=(1, 1, 1))
        self.bn8 = nn.BatchNorm3d(128)
        self.trans_conv4 = nn.ConvTranspose3d(in_channels=256, out_channels=64, kernel_size=(4, 4, 4), stride=(2, 2, 2),
                                              padding=(1, 1, 1))
        self.bn9 = nn.BatchNorm3d(64)
        self.trans_conv5 = nn.ConvTranspose3d(in_channels=128, out_channels=32, kernel_size=(4, 4, 4), stride=(2, 2, 2),
                                              padding=(1, 1, 1))
        self.bn10 = nn.BatchNorm3d(32)
        self.trans_conv6 = nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size=(4, 4, 4), stride=(2, 2, 2),
                                              padding=(1, 1, 1))

        self.activation_function = nn.Tanh()
        initialize_weights(self)

    def forward(self, x):
        # down sampling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x2 = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x3 = x
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x4 = x
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x5 = x
        x = self.conv6(x)

        # up-sampling
        x = self.trans_conv1(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = torch.concat([x, x5], dim=1)

        x = self.trans_conv2(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = torch.concat([x, x4], dim=1)

        x = self.trans_conv3(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = torch.concat([x, x3], dim=1)

        x = self.trans_conv4(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = torch.concat([x, x2], dim=1)

        x = self.trans_conv5(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = torch.concat([x, x1], dim=1)

        x = self.trans_conv6(x)
        x = self.activation_function(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.dim = dim

        self.image_to_features = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=dim, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=dim, out_channels=2 * dim, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=2 * dim, out_channels=4 * dim, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=4 * dim, out_channels=8 * dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=8 * dim, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.image_to_features(x)
        return x