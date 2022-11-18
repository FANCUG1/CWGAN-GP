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
                 print_every=50, use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'con_loss': [],
                       'total_G': []}
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

        g_loss = - d_generated.mean()
        con_loss = self.section_loss_func2(data, label, fake_label)
        G_total_loss = g_loss + 100 * con_loss
        G_total_loss.backward()
        self.G_opt.step()

        self.losses['G'].append(g_loss.item())
        self.losses['con_loss'].append(con_loss.item())
        self.losses['total_G'].append(G_total_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        real_data = real_data.detach()
        generated_data = generated_data.detach()

        alpha = torch.rand(batch_size, 1, 1, 1, 1)
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
                    print("con_loss: %.8f" % self.losses['con_loss'][-1])
                    print("G_total_loss: %.4f" % self.losses['total_G'][-1])

    def train(self, data_loader, epochs):

        sample, sample_label = data_loader.dataset[0]

        sample_label = sample_label.detach().cpu().numpy()
        sample_label = sample_label.reshape((64, 64, 64))
        sample_label = sample_label.reshape((64 * 64 * 64, 1))  # z,y,x
        np.savetxt('./fake_images/sample_real.txt', sample_label, fmt='%1.2f')

        sample = Variable(sample.view(1, 1, 64, 64, 64)).cuda()

        for epoch in range(epochs):

            print("\n Epoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

            self.G.eval()
            generated_data = self.G(sample)

            generated_data = generated_data.detach().cpu().numpy()
            generated_data = generated_data.reshape((64, 64, 64))
            generated_data = generated_data.reshape((64 * 64 * 64, 1))

            for i in range(len(generated_data)):
                if generated_data[i] < 0:
                    generated_data[i] = -1
                else:
                    generated_data[i] = 1

            np.savetxt('./fake_images/generated_images_{}.txt'.format(epoch + 1), generated_data, fmt='%1.2f')

            torch.save(self.G.state_dict(), './WGAN_GP_Model_3D/WGAN_GP_3D_Generator_{}.pth'.format(epoch + 1))
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

    def free_params(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = True

    def frozen_params(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def section_loss_func2(self, data, real_label, fake_label):
        bs = data.size()[0]
        data = data.detach().cpu().numpy()
        real_label = real_label.detach().cpu().numpy()
        fake_label = fake_label.detach().cpu().numpy()
        for i in range(bs):
            temp1 = data[i].reshape((64, 64, 64))
            temp2 = real_label[i].reshape((64, 64, 64))
            temp3 = fake_label[i].reshape((64, 64, 64))
            z, y, x = temp1.shape
            loss_Y_X_Direction = 0.0
            loss_Z_X_Direction = 0.0
            loss_Z_Y_Direction = 0.0

            # find Z-Y cross section
            for j in range(x):
                if (temp1[:, :, j] == temp2[:, :, j]).all():
                    loss_Z_Y_Direction = ((temp3[:, :, j] - temp2[:, :, j]) ** 2).mean()  # L2
                    # loss_Z_Y_Direction = (abs(temp3[:, :, j] - temp2[:, :, j])).mean()  # L1

            # find Z-X cross section
            for j in range(y):
                if (temp1[:, j, :] == temp2[:, j, :]).all():
                    loss_Z_X_Direction = ((temp3[:, j, :] - temp2[:, j, :]) ** 2).mean()  # L2
                    # loss_Z_X_Direction = (abs(temp3[:, j, :] - temp2[:, j, :])).mean()  # L1

            # find Y-X cross section
            for j in range(z):
                if (temp1[j, :, :] == temp2[j, :, :]).all():
                    loss_Y_X_Direction = ((temp3[j, :, :] - temp2[j, :, :]) ** 2).mean()  # L2
                    # loss_Y_X_Direction = (abs(temp3[j, :, :] - temp2[j, :, :])).mean()  # L1

            total_loss = loss_Y_X_Direction + loss_Z_Y_Direction + loss_Z_X_Direction
            total_loss = np.array(total_loss, dtype=np.float16)
            total_loss = torch.from_numpy(total_loss).type(torch.FloatTensor)
            return Variable(total_loss).cuda()
