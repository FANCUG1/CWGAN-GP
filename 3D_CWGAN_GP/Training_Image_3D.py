import torch
import torch.nn as nn
import os
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import cv2


def total_image_prepare():
    array = np.loadtxt('./test_data.txt', dtype=np.float16)
    array = (array - 0.5) / 0.5
    array = array.reshape((120, 150, 180))
    list = []

    for i in range(0, 120, 5):  # z-axis
        for j in range(0, 150, 5):  # y-axis
            for k in range(0, 180, 5):  # x-axis
                if (i + 64) < 120 and (j + 64) < 150 and (k + 64) < 180:
                    temp = array[i:i + 64, j:j + 64, k:k + 64]
                    list.append(temp.astype(np.float16))

    return list


# prepare training dataset and testing dataset
def dataset_prepare():
    total_dataset = total_image_prepare()
    count = 0
    for data in total_dataset:
        data = data.reshape((64 * 64 * 64, -1))
        np.savetxt('./dataset3D/training_dataset_{}.txt'.format(count), data, fmt='%1.0f')
        count = count + 1


# choose cross section for each 3D subsurface model
def add_cross_section(data, label):
    size = data.shape[0]
    # select location
    location = random.randint(0, size - 1)  # select a location randomly

    # x-y cross section
    for i in range(size):
        for j in range(size):
            data[location][i][j] = label[location][i][j]

    # x-z cross section
    for i in range(size):
        for j in range(size):
            data[i][location][j] = label[i][location][j]

    # y-z cross section
    for i in range(size):
        for j in range(size):
            data[i][j][location] = label[i][j][location]

    return data


def training_image_prepare():
    total_data_list = total_image_prepare()
    training_image_label = []  # label for training dataset
    testing_image_label = []  # label for test datset
    count = 0
    for data in total_data_list:
        if count < 5000:
            training_image_label.append(data)
            count += 1
        else:
            testing_image_label.append(data)

    training_data = []
    testing_data = []
    for i in range(len(training_image_label)):
        temp1 = np.zeros((64, 64, 64), dtype=np.float16)
        temp2 = training_image_label[i]
        data_with_cross_section = add_cross_section(temp1, temp2)
        training_data.append(data_with_cross_section)

    for i in range(len(testing_image_label)):
        temp1 = np.zeros((64, 64, 64), dtype=np.float16)
        temp2 = testing_image_label[i]
        data_with_cross_section = add_cross_section(temp1, temp2)
        testing_data.append(data_with_cross_section)

    training_image_label = np.array(training_image_label, dtype=np.float16)
    training_image_label = training_image_label.reshape((len(training_image_label), -1, 64, 64, 64))
    training_data = np.array(training_data, dtype=np.float16)
    training_data = training_data.reshape((len(training_data), -1, 64, 64, 64))

    testing_image_label = np.array(testing_image_label, dtype=np.float16)
    testing_image_label = testing_image_label.reshape((len(testing_image_label), -1, 64, 64, 64))
    testing_data = np.array(testing_data, dtype=np.float16)
    testing_data = testing_data.reshape((len(testing_data), -1, 64, 64, 64))

    temp1 = torch.FloatTensor(training_image_label)
    temp2 = torch.FloatTensor(training_data)
    train_dataset = TensorDataset(temp2, temp1)

    temp1 = torch.FloatTensor(testing_image_label)
    temp2 = torch.FloatTensor(testing_data)
    test_dataset = TensorDataset(temp2, temp1)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
    return train_dataloader, test_dataloader




