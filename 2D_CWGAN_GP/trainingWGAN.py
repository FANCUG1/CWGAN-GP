import math
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math


class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, gp_weight=10, critic_iterations=5,
                 print_every=100, use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'con_loss': [],
                       'total_G': []}  # record each loss
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data, label):
        self.D.train()
        self.frozen_params(self.G)
        self.free_params(self.D)

        fake_label = self.G(data)

        real_batch = torch.concat([data, label], dim=1)  # bs 2 64 64
        fake_batch = torch.concat([data, fake_label], dim=1)  # bs 2 64 64

        d_real_image_score = self.D(real_batch)
        d_rec_image_score = self.D(fake_batch)

        gradient_penalty = self._gradient_penalty(real_batch, fake_batch)
        self.losses['GP'].append(gradient_penalty.item())

        self.D_opt.zero_grad()
        d_loss = d_rec_image_score.mean() - d_real_image_score.mean() + gradient_penalty
        d_loss.backward()
        self.D_opt.step()

        self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, data, label):
        self.G.train()
        self.frozen_params(self.D)
        self.free_params(self.G)

        self.G_opt.zero_grad()

        fake_label = self.G(data)

        temp_D = torch.concat([data, fake_label], dim=1)

        d_generated = self.D(temp_D)

        con_loss = self.conditional_loss_func(data, label, fake_label)
        g_loss = - d_generated.mean()
        G_Total_loss = g_loss + con_loss * 1000
        G_Total_loss.backward()
        self.G_opt.step()

        self.losses['G'].append(g_loss.item())
        self.losses['con_loss'].append(con_loss.item())
        self.losses['total_G'].append(G_Total_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        real_data = real_data.detach()
        generated_data = generated_data.detach()

        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)

        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        prob_interpolated = self.D(interpolated)

        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(
            prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, (data, label) in enumerate(data_loader, 0):
            bs = data.size()[0]
            self.num_steps += 1

            data = Variable(data).cuda()
            label = Variable(label).cuda()

            self._critic_train_iteration(data, label)

            if self.num_steps % 5 == 0:
                self._generator_train_iteration(data, label)

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: %.4f" % self.losses['D'][-1])
                print("GP: %.4f" % self.losses['GP'][-1])
                print("Gradient norm: %.4f" % self.losses['gradient_norm'][-1])
                if self.num_steps > self.critic_iterations:
                    print("G_origin: %.4f" % self.losses['G'][-1])
                    print("con_loss: %.8f" % self.losses["con_loss"][-1])
                    print("G_total_loss: %.4f" % self.losses["total_G"][-1])

    def train(self, data_loader, epochs):

        sample, sample_label = data_loader.dataset[0]
        save_image(sample_label, './fake_images/sample_real.jpeg')
        sample = Variable(sample.view(1, 1, 64, 64)).cuda()
        con_point = self.draw_condition(sample)
        save_image(con_point, './fake_images/sample_con_point.jpeg')

        for epoch in range(epochs):
            print("\n Epoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

            self.G.eval()
            generated_data = self.G(sample)

            img_grid = make_grid(generated_data.detach().cpu())
            save_image(img_grid, './fake_images/fake_images_{}.png'.format(epoch + 1))

            # 每5轮进行一次模型的保存
            torch.save(self.G.state_dict(), './WGAN_GP_Model/WGAN_GP_Generator_{}.pth'.format(epoch + 1))
            # torch.save(self.D.state_dict(), './WGAN_GP_Model/WGAN_GP_Discriminator_{}.pth'.format(epoch + 1))

    def plot_loss(self, list1, list2):
        plt.figure(figsize=(10, 5))
        plt.title("Loss Functions G & D")
        plt.plot(list1)
        plt.plot(list2)
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.legend(['Generator_loss', 'Discriminator_loss'])
        plt.show()

    # record conditioning data as 1, unconditioning data as 0
    def mask_operation(self, data):
        row, column = data.shape
        binary_mask = np.zeros((column, row))
        for i in range(column):
            for j in range(row):
                if data[i][j] == -1 or data[i][j] == 1:
                    binary_mask[i][j] = 1
        return binary_mask

    def expansion_func(self, data, x_location, y_location):
        if (x_location - 1) >= 0 and (y_location - 1) >= 0:
            data[x_location - 1][y_location - 1] = 1
        if (x_location - 1) >= 0 and y_location >= 0:
            data[x_location - 1][y_location] = 1
        if (x_location - 1) >= 0 and (y_location + 1) <= 63:
            data[x_location - 1][y_location + 1] = 1
        if x_location >= 0 and (y_location - 1) >= 0:
            data[x_location][y_location - 1] = 1
        if x_location >= 0 and (y_location + 1) <= 63:
            data[x_location][y_location + 1] = 1
        if (x_location + 1) <= 63 and (y_location - 1) >= 0:
            data[x_location + 1][y_location - 1] = 1
        if (x_location + 1) <= 63 and y_location <= 63:
            data[x_location + 1][y_location] = 1
        if (x_location + 1) <= 63 and (y_location + 1) <= 63:
            data[x_location + 1][y_location + 1] = 1
        return data

    def binary_mask_expansion(self, data):
        column, row = data.shape
        temp = data
        for i in range(column):
            for j in range(row):
                if temp[i][j] == 1:
                    data = self.expansion_func(data, i, j)
        return data

    def binary_mask_tensor(self, data):

        bs = data.size()[0]
        data = data.detach().cpu().numpy()
        mask_list1 = []  # conditioning data without expansion
        mask_list2 = []  # conditioning data with expansion
        for i in range(bs):
            temp = data[i].reshape((64, 64))
            binary_mask = self.mask_operation(temp)
            mask_list1.append(binary_mask.reshape((1, 64, 64)))

            binary_mask_expansion = self.binary_mask_expansion(binary_mask)
            mask_list2.append(binary_mask_expansion.reshape((1, 64, 64)))

        mask_list1 = np.array(mask_list1)
        mask_list2 = np.array(mask_list2)
        mask_tensor_without_expansion = torch.from_numpy(mask_list1).type(torch.FloatTensor)
        mask_tensor_with_expansion = torch.from_numpy(mask_list2).type(torch.FloatTensor)

        return Variable(mask_tensor_without_expansion).cuda(), Variable(mask_tensor_with_expansion).cuda()

    def conditional_loss_func(self, data, real_label, fake_label):
        mask_tensor, mask_tensor_expansion = self.binary_mask_tensor(data)
        context_loss = torch.mean(mask_tensor_expansion * torch.abs(fake_label - real_label))
        return context_loss

    def free_params(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    def frozen_params(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def draw_condition(self, data):
        bs = data.size()[0]
        data = data.detach().cpu().numpy()
        for i in range(bs):
            temp = data[i].reshape((64, 64))
            for j in range(64):
                for k in range(64):
                    if temp[j][k] != -1 and temp[j][k] != 1:
                        temp[j][k] = 0
                    else:
                        temp[j][k] = 255
            data[i] = temp.reshape((1, 64, 64))
        return torch.from_numpy(data)
