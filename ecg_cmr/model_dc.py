import torch
from torch import nn
# import tensorflow as tf
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import wandb
import numpy as np




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Z_DIM = 100
        # self.fc = nn.Linear(12 * 5000 + Z_DIM, 512 * 5 * 5)

        self.fc = nn.Linear(12*5000, 512*5*5)

        self.main = nn.Sequential(
            # 输入尺寸: (512, 5, 5)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 尺寸: (256, 10, 10)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 尺寸: (128, 20, 20)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 尺寸: (64, 40, 40)
            nn.ConvTranspose2d(64, 50, 4, 2, 1, bias=False),
            # 输出尺寸: (50, 80, 80)
        )

    # def forward(self, input):
    #     # condition = condition.view(condition.size(0), -1)  # 展平条件数据
    #     # input = z + condition.view(z.shape[0], -1)
    #     # input = torch.cat((z, condition), dim=1)  # 将噪声向量和条件数据一起输入
    #     fc_output = self.fc(input)
    def forward(self, input):
        input = input.view(input.size(0), -1)  # 展开输入
        fc_output = self.fc(input)
        fc_output = fc_output.view(input.size(0), 512, 5, 5)  # 将全连接层输出调整为所需的二维形状
        return self.main(fc_output)


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             # 输入是一个尺寸为 (1,12,5000) 的张量
#             # 假设输入内容应首先被调整为适合的形状以适应3D卷积层
#             nn.ConvTranspose2d(1, 512, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             # 尺寸: (512, 4, 4)
#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             # 尺寸: (256, 8, 8)
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             # 尺寸: (128, 16, 16)
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             # 尺寸: (64, 32, 32)
#             nn.ConvTranspose2d(64, 50, 4, 2, 1, bias=False),
#             # 输出尺寸: (50, 80, 80)
#         )
#
#     def forward(self, input):
#         return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.ecg_branch = nn.Sequential(
            nn.Conv1d(1, 32, 10, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, 10, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 10, stride=2),
            nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((80, 80))  # 将ECG数据最后一个维度转化为 (80, 80)

        self.main = nn.Sequential(
            nn.Conv2d(178, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, ecg):
        ecg = ecg.view(ecg.shape[0], 1, -1)
        ecg = self.ecg_branch(ecg)
        ecg = ecg.unsqueeze(-1)

        ecg = self.adaptive_pool(ecg)

        x = torch.cat((x, ecg), dim=1)

        return self.main(x)
    # def forward(self, x, condition):
    #
    #     # 创建一个新的全零张量，width是condition.shape[3]，其他维度与x保持一致
    #     new_condition = torch.zeros(x.size(0), 50, x.size(2), condition.shape[3]).to(x.device)
    #
    #     # 将condition的值复制到new_condition的对应位置
    #     new_condition[:, :, :condition.shape[2], :condition.shape[3]] = condition
    #
    #     x = torch.cat((x, new_condition), dim=1)  # 将核磁图像和心电图数据合并
    #     return self.main(x)
    # def forward(self, x, condition):
    #     # condition = condition.unsqueeze(1)
    #
    #     new_condition = torch.zeros(x.size(0), 50, 80, 80).to(x.device)
    #
    #     # 将 condition 的值复制到 new_condition 的合适位置
    #     new_condition[:, :, :condition.shape[2], :condition.shape[3]] = condition
    #     # condition = condition.expand(-1, -1, x.shape[2], x.shape[3])
    #     x = torch.cat((x, new_condition), dim=1)  # 将核磁图像和心电图数据合并
    #     return self.main(x)
        # return self.main(input)


class GANModel(nn.Module):
    def __init__(self):
        super(GANModel, self).__init__()
        # LAMBDA = 100
        # loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = Generator()
        self.discriminator = Discriminator()

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x, ecg):
        return self.discriminator(x, ecg)

    def generator_loss(self, disc_generated_output, gen_output, target):
        LAMBDA = 100
        loss_object = nn.BCEWithLogitsLoss()

        gan_loss = loss_object(disc_generated_output, torch.ones_like(disc_generated_output))
        # print(f"target.shape:{target.shape}")
        # print(f"gen_output.shape:{gen_output.shape}")
        # Mean absolute error
        l1_loss = torch.mean(torch.abs(target - gen_output))

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss


    def discriminator_loss(self, disc_real_output, disc_generated_output):
        loss_object = nn.BCEWithLogitsLoss()

        real_loss = loss_object(disc_real_output, torch.ones_like(disc_real_output).to(disc_real_output.device))
        generated_loss = loss_object(disc_generated_output,
                                     torch.zeros_like(disc_generated_output).to(disc_generated_output.device))

        total_disc_loss = real_loss + generated_loss
        return total_disc_loss


    def train(self, train_loader, num_epochs=400, learning_rate=0.00002):
        torch.autograd.set_detect_anomaly(True)

        optim_g = optim.Adam(self.generator.parameters(), lr=learning_rate)
        optim_d = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        save_dir = "../data_ecg_cmr/checkpoints_dc/"
        data_dir = "./data/"

        for epoch in range(num_epochs):
            for i, (ecg_data, cmr_real_data) in enumerate(train_loader):
                # 对输入数据和真实数据进行处理
                ecg_data = ecg_data.to(device)
                cmr_real_data = cmr_real_data.to(device)
                Z_DIM = 60000
                batch_size = 16
                z = torch.randn((batch_size, Z_DIM), device=device)
                # print(f"ecg_data:{ecg_data.shape}")

                # 训练判别器
                optim_d.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    real_outputs = self.discriminator(cmr_real_data, ecg_data)
                    # print(f"real_outputs:{real_outputs.shape}")
                    # real_loss = self.discriminator_loss(real_outputs, torch.ones_like(real_outputs).to(device))
                    # print(f"real_loss:{real_loss.shape}")
                    cmr_fake_data = self.generator(ecg_data)
                    # print(f"cmr_fake_data:{cmr_fake_data.shape}")

                    fake_outputs = self.discriminator(cmr_fake_data, ecg_data)
                    # print(f"fake_outputs:{fake_outputs.shape}")
                disc_loss = self.discriminator_loss(real_outputs, fake_outputs)

                # disc_loss = (real_loss + fake_loss) / 2

                disc_loss.backward(retain_graph=True)
                optim_d.step()

                # 训练生成器
                optim_g.zero_grad()
                cmr_fake_data = self.generator(ecg_data)
                # print(f"cmr_fake_data:{cmr_fake_data.shape}")

                fake_outputs = self.discriminator(cmr_fake_data, ecg_data)
                # with torch.autograd.set_detect_anomaly(True):
                    # fake_outputs_g = self.generator(ecg_data)
                    # print(f"fake_outputs——g:{fake_outputs.shape}")
                gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(fake_outputs, cmr_fake_data, cmr_real_data)
                gen_total_loss.backward(retain_graph=True)
                optim_g.step()

                # 输出训练信息
                if i % 8 == 0:
                    print("Epoch: {}, Iter: {}, D loss: {:.4f}, G-total loss: {:.4f}, G-gan loss: {:.4f}, G-l1 loss: {:.4f}".format(epoch, i, disc_loss.item(),
                                                                                       gen_total_loss.item(), gen_gan_loss.item(), gen_l1_loss.item()))
                    wandb.log(
                        {"Epoch": epoch, "Iter": i, "D loss": disc_loss.item(), "G-total loss": gen_total_loss.item(),
                         "G-gan loss": gen_gan_loss.item(), "G-l1 loss": gen_l1_loss.item()})
                    # torch.save(self.generator.state_dict(), os.path.join(save_dir, f'generator_epoch_{epoch}.pt'))
                    # torch.save(self.discriminator.state_dict(),
                    #            os.path.join(save_dir, f'discriminator_epoch_{epoch}.pt'))

                # wandb.log({"Epoch: {}, Iter: {}, D loss: {:.4f}, G-total loss: {:.4f}, G-gan loss: {:.4f}, G-l1 loss: {:.4f}".format(epoch, i, disc_loss.item(),
                #                                                                            gen_total_loss.item(), gen_gan_loss.item(), gen_l1_loss.item())})
            if epoch > 180:
                # print(0)
                torch.save(self.generator.state_dict(), os.path.join(save_dir, f'generator5_epoch_{epoch}.pt'))
                torch.save(self.discriminator.state_dict(), os.path.join(save_dir, f'discriminator5_epoch_{epoch}.pt'))

        # with torch.no_grad():
        #     z = torch.randn(64, 100).to(device)
        #     fake_data = self.generator(z).detach().cpu()
            # torch.save(fake_data, os.path.join(data_dir, 'fake_data.pt'))

class CustomDataset(Dataset):
    def __init__(self, data_path):
        data_dir = torch.load(data_path)
        keys = list(data_dir.keys())
        self.ecg_data = data_dir[keys[0]]  # 输入数据
        self.cmr_data = data_dir[keys[1]]  # 真实的生成数据
        self.ecg_data = self.ecg_data.unsqueeze(1)
        self.ecg_data = np.array(self.ecg_data)
        self.cmr_data = np.array(self.cmr_data)

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        return self.ecg_data[idx], self.cmr_data[idx]


run = wandb.init(project='ecg_cmr')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # 设备设定
data_path = "../../data/ECG_CMR/train_data_dict_v7.pt"
dataset = CustomDataset(data_path)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

gan = GANModel().to(device)
wandb.watch(gan)
gan.train(train_loader=dataloader, num_epochs=400, learning_rate=0.00002)  # 训练模型

wandb.finish()



    # def train_step(input_image, target, step):
    #
    #     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    #         gen_output = generator(input_image, training=True)
    #
    #         disc_real_output = discriminator([input_image, target], training=True)
    #         disc_generated_output = discriminator([input_image, gen_output], training=True)
    #
    #         gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    #         disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    #
    #     generator_gradients = gen_tape.gradient(gen_total_loss,
    #                                             generator.trainable_variables)
    #     discriminator_gradients = disc_tape.gradient(disc_loss,
    #                                                  discriminator.trainable_variables)
    #
    #     generator_optimizer.apply_gradients(zip(generator_gradients,
    #                                             generator.trainable_variables))
    #     discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
    #                                                 discriminator.trainable_variables))
    #
    #     with summary_writer.as_default():
    #         tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
    #         tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
    #         tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
    #         tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)
    #
    # def fit(train_ds, test_ds, steps):
    #     example_input, example_target = next(iter(test_ds.take(1)))
    #     start = time.time()
    #
    #     for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    #         if (step) % 1000 == 0:
    #             display.clear_output(wait=True)
    #
    #             if step != 0:
    #                 print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')
    #
    #             start = time.time()
    #
    #             generate_images(generator, example_input, example_target)
    #             print(f"Step: {step // 1000}k")
    #
    #         train_step(input_image, target, step)
    #
    #         # Training step
    #         if (step + 1) % 10 == 0:
    #             print('.', end='', flush=True)
    #
    #         # Save (checkpoint) the model every 5k steps
    #         if (step + 1) % 5000 == 0:
    #             checkpoint.save(file_prefix=checkpoint_prefix)