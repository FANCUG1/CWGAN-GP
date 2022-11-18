import numpy as np
from torch.autograd import Variable

from Training_Image_3D import training_image_prepare
from WGAN3D_model import Generator
import torch
import torch.nn as nn
import os


def array_handle(array):
    for i in range(len(array)):
        if array[i] < 0:
            array[i] = -1
        else:
            array[i] = 1
    return array


_, test_dataset = training_image_prepare()

model_gen = Generator().cuda()
model_gen.load_state_dict(torch.load('./WGAN_GP_Model_3D/WGAN_GP_3D_Generator_160.pth'))
model_gen.eval()
num1 = 0  # record label
num2 = 0  # record result
num3 = 0  # record section data information


with torch.no_grad():
    for i, (datav, label) in enumerate(test_dataset, 0):
        bs = datav.size()[0]

        numpy_label = label.detach().cpu().numpy()
        for j in range(bs):
            temp_label = numpy_label[j].reshape((64 * 64 * 64, 1))  # z,y,x
            np.savetxt('./test_real_data/real_image_{}.txt'.format(num1 + 1), temp_label, fmt='%1.2f')
            num1 += 1

        datav = Variable(datav).cuda()
        result = model_gen(datav)
        result = result.detach().cpu().numpy()

        for k in range(bs):
            temp_result = result[k].reshape((64 * 64 * 64, 1))

            for s in range(len(temp_result)):
                if temp_result[s] < 0:
                    temp_result[s] = -1
                else:
                    temp_result[s] = 1

            np.savetxt('./test_generated_data/generated_image_{}.txt'.format(num2 + 1), temp_result, fmt='%1.2f')
            num2 += 1


        datav = datav.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        for j in range(bs):
            os.mkdir('./test_section_data/test_data_{}_section_information'.format(num3 + 1))
            temp1 = datav[j].reshape((64, 64, 64))  # z y x
            temp2 = label[j].reshape((64, 64, 64))  # z y x
            temp3 = result[j].reshape((64, 64, 64))  # z y x
            z, y, x = temp1.shape

            # Z-Y cross section
            for k in range(x):
                if (temp1[:, :, k] == temp2[:, :, k]).all():
                    array = temp3[:, :, k].reshape((64 * 64, 1))
                    array = array_handle(array)

                    real = temp2[:, :, k].reshape((64 * 64, 1))
                    # 1 64 64
                    np.savetxt(
                        './test_section_data/test_data_{}_section_information/section_ZY_for_generated_data_in_location_{}.txt'.format(
                            num3 + 1, k), array, fmt='%1.2f')
                    np.savetxt(
                        './test_section_data/test_data_{}_section_information/section_ZY_for_real_data_in_location_{}.txt'.format(
                            num3 + 1, k), real, fmt='%1.2f')

            # Z-X cross section
            for k in range(y):
                if (temp1[:, k, :] == temp2[:, k, :]).all():
                    array = temp3[:, k, :].reshape((64 * 64, 1))
                    array = array_handle(array)
                    real = temp2[:, k, :].reshape((64 * 64, 1))
                    # 64 1 64
                    np.savetxt(
                        './test_section_data/test_data_{}_section_information/section_ZX_for_generated_data_in_location_{}.txt'.format(
                            num3 + 1, k), array, fmt='%1.2f')
                    np.savetxt(
                        './test_section_data/test_data_{}_section_information/section_ZX_for_real_data_in_location_{}.txt'.format(
                            num3 + 1, k), real, fmt='%1.2f')

            # Y-X cross section
            for k in range(x):
                if (temp1[k, :, :] == temp2[k, :, :]).all():
                    array = temp3[k, :, :].reshape((64 * 64, 1))
                    array = array_handle(array)
                    real = temp2[k, :, :].reshape((64 * 64, 1))
                    # 64 64 1
                    np.savetxt(
                        './test_section_data/test_data_{}_section_information/section_YX_for_generated_data_in_location_{}.txt'.format(
                            num3 + 1, k), array, fmt='%1.2f')
                    np.savetxt(
                        './test_section_data/test_data_{}_section_information/section_YX_for_real_data_in_location_{}.txt'.format(
                            num3 + 1, k), real, fmt='%1.2f')
            num3 += 1
