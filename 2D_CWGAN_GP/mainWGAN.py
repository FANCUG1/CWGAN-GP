import random
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from WGAN_model import Generator, Discriminator
from trainingWGAN import Trainer
from Training_Image import training_image_dataloader
import cv2

train_dataloader, test_dataloader = training_image_dataloader()

generator = Generator()
discriminator = Discriminator(dim=32)

G_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 50
generator.train()
discriminator.train()
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, use_cuda=torch.cuda.is_available())
trainer.train(train_dataloader, epochs)

trainer.plot_loss(trainer.losses['total_G'], trainer.losses['D'])

'''
# validation process
model_gen = Generator().cuda()
model_gen.load_state_dict(torch.load('./WGAN_GP_Model/WGAN_GP_Generator.pth'))
model_gen.eval()
num = 0
count_real = 0
cnt = 0
count = 0

with torch.no_grad():
    for i, (datav, label) in enumerate(test_dataloader, 0):
        bs = datav.size()[0]
        save_image(label, './fake_images/model_training_test/real_image_{}.png'.format(i + 1))
        # temp_list = []
        datav = datav.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        for temp_array in label:
            temp_array = temp_array.reshape((64, 64))
            # temp_array = temp_array.reshape((64 * 64, 1))
            array = np.zeros((64, 64))
            for i1 in range(64):
                array[63 - i1, :] = temp_array[i1, :]
            array = array.reshape((64 * 64, 1))
        
            np.savetxt('./test_real_data/real_value_{}.txt'.format(count_real + 1), array, fmt='%1.2f')
            count_real += 1

        for array2 in datav:
            temp = array2.reshape((64, 64))
            temp_zeros = np.zeros((64, 64))
            for i2 in range(64):
                for j2 in range(64):
                    if temp[i2][j2] == -1 or temp[i2][j2] == 1:
                        temp_zeros[i2][j2] = 255
            cv2.imwrite('./Conditional_information/conditional_picture/conditional_point_{}.png'.format(count + 1), temp_zeros)
            count += 1

        datav = torch.from_numpy(datav)
        datav = Variable(datav).cuda()

        result = model_gen(datav)
        result = result.detach()
        
        save_image(result, './fake_images/model_training_test/rec_image_{}.png'.format(i + 1))

        result = result.detach().cpu().numpy()
  
        for temp2 in result:
            temp2 = temp2.reshape((64, 64))
            array = np.zeros((64, 64))
            for i2 in range(64):
                array[63 - i2, :] = temp2[i2, :]
            array = temp2.reshape((64 * 64, 1))
  
            for j in range(len(array)):
                if array[j] - int(array[j]) >= 0.5:
                    array[j] = int(array[j]) + 1

                else:
                    array[j] = int(array[j])
                    if array[j] == -1:
                        array[j] = array[j] + 1
            np.savetxt('./test_final_results/test_{}.txt'.format(num + 1), array, fmt='%1.2f')
            num += 1
'''
