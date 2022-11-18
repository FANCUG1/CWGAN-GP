import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import cv2
import random


# prepare 2D subsurface model
def training_image_prepare():
    temp = np.loadtxt('text_data.txt').astype(int)
    temp = temp.reshape(250, 250)
    array = np.zeros((250, 250))
    for i in range(250):
        array[249 - i, :] = temp[i, :]

    count = 0
    list = []

    # sliding-window-based segmentation
    for i in range(0, 186, 1):
        for j in range(0, 186, 1):
            temp = array[i:i + 64, j:j + 64]
            list.append(temp)

    for data in list:
        cv2.imwrite('./Training_Image_Set' + str(count) + '.png', cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
        count += 1


def training_image_dataloader():
    list_training_label = []
    list_testing_label = []

    for i in range(0, 34596, 1):
        src = cv2.imread('./Training_Image_Set/' + str(i) + '.png', flags=0)
        src = src.reshape((1, 64, 64))
        src = (src - 127.5) / 127.5  # normalize between [-1,1]

        if i < 34000:
            list_training_label.append(src)  # training dataset label
        else:
            list_testing_label.append(src)  # test dataset label

    list_training_data = []
    list_testing_data = []

    # prepare training datset conditioning data
    for array in list_training_label:
        array = array.reshape((64, 64))
        temp = np.zeros((64, 64))
        count_0 = 0
        count_1 = 0

        while count_0 + count_1 < 12:
            j = random.randint(0, 63)
            k = random.randint(0, 63)
            if array[j][k] == -1 and temp[j][k] != -1 and temp[j][k] != 1 and count_0 < 6:
                temp[j][k] = array[j][k]
                count_0 += 1
            elif array[j][k] == 1 and temp[j][k] != -1 and temp[j][k] != 1 and count_1 < 6:
                temp[j][k] = array[j][k]
                count_1 += 1
        temp = temp.reshape((1, 64, 64))
        list_training_data.append(temp)

    # preparing testing_dataset_conditioning data
    for array in list_testing_label:
        array = array.reshape((64, 64))
        temp = np.zeros((64, 64))
        count_0 = 0
        count_1 = 0

        while count_0 + count_1 < 12:
            j = random.randint(0, 63)
            k = random.randint(0, 63)
            if array[j][k] == -1 and temp[j][k] != -1 and temp[j][k] != 1 and count_0 < 6:
                temp[j][k] = array[j][k]
                count_0 += 1
            elif array[j][k] == 1 and temp[j][k] != -1 and temp[j][k] != 1 and count_1 < 6:
                temp[j][k] = array[j][k]
                count_1 += 1
        temp = temp.reshape((1, 64, 64))
        list_testing_data.append(temp)

    list_training_image = np.array(list_training_data)
    list_testing_image = np.array(list_testing_data)
    training_label = np.array(list_training_label)
    testing_label = np.array(list_testing_label)

    list_training_image = list_training_image.reshape((len(list_training_image), -1, 64, 64))
    training_label = training_label.reshape((len(training_label), -1, 64, 64))
    list_testing_image = list_testing_image.reshape((len(list_testing_image), -1, 64, 64))
    testing_label = testing_label.reshape((len(testing_label), -1, 64, 64))

    train = torch.FloatTensor(list_training_image)
    label_train = torch.FloatTensor(training_label)
    test = torch.FloatTensor(list_testing_image)
    label_test = torch.FloatTensor(testing_label)

    train_dataset = TensorDataset(train, label_train)
    test_dataset = TensorDataset(test, label_test)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    training_image_prepare()


