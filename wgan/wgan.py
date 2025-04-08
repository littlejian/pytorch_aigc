import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision

# 超参数
batch_size = 64
lr = 0.0001
beta1 = 0.5
beta2 = 0.999
epochs = 100
latent_dim = 100
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

# 输入图像尺寸
image_size = 64
channels = 3

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)  # 隐藏层，输出 4x4x256 的特征图
        self.conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 转置卷积层，输出 8x8x128
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 转置卷积层，输出 16x16x64
        self.conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 转置卷积层，输出 32x32x32
        self.conv4 = nn.ConvTranspose2d(32, channels, 4, 2, 1)  # 输出 64x64x3 的图像

    def forward(self, z):
        x = F.relu(self.fc(z)).view(-1, 256, 4, 4)  # 通过全连接层然后 reshape
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.tanh(self.conv4(x))  # 使用tanh激活输出图像
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 4, 2, 1)  # 输入 64x64x3 图像，输出 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)         # 输出 16x16x64
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)        # 输出 8x8x128
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)       # 输出 4x4x256
        self.fc = nn.Linear(256 * 4 * 4, 1)              # 最终输出一个标量

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc(x)
        return x

def compute_gradient_penalty(D, real_samples, fake_samples):
    # 获取batch的大小
    batch_size = real_samples.size(0)
    # 在 [0, 1] 范围内进行插值
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated_images = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated_images.requires_grad_(True)

    # 计算判别器输出
    d_interpolated = D(interpolated_images)

    # 计算梯度
    gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated_images,
                                    grad_outputs=torch.ones_like(d_interpolated).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    # 计算梯度的L2范数
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# 模型实例化
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# 优化器
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# 数据加载器
transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_data = torchvision.datasets.ImageFolder(root='./files/anime', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

def save_generated_images(epoch, generator, latent_dim, save_dir='./generated_images/'):
    """保存生成的图像并可视化"""
    z = torch.randn(32, latent_dim).to(device)  # 生成16个随机噪声
    generated_images = generator(z)  # 生成图像
    generated_images = generated_images.cpu().detach()  # 转到CPU并分离图像张量

    # 将图像转换为适合显示的形式
    grid = torchvision.utils.make_grid(generated_images, nrow=16, normalize=True, padding=2)
    
    # 显示图像
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f"Generated Images at Epoch {epoch}")
    plt.show()

    # 保存图像
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.imsave(f"{save_dir}/generated_epoch_{epoch}.png", grid.permute(1, 2, 0).numpy())

# 训练循环
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)

        # 训练判别器
        optimizer_D.zero_grad()

        # 生成假图像
        z = torch.randn(real_images.size(0), latent_dim).to(device)
        fake_images = generator(z)

        # 判别器对真实图像和假图像的判别
        real_validity = discriminator(real_images)
        fake_validity = discriminator(fake_images.detach())

        # 计算梯度惩罚
        gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images)

        # 判别器损失
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
        d_loss.backward(retain_graph=True)
        optimizer_D.step()

        # 训练生成器
        if i % 5 == 0:
            optimizer_G.zero_grad()

            # 生成器损失
            fake_validity = discriminator(fake_images)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()

    # 每5个epoch可视化生成图像
    if epoch % 1 == 0:
        print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
        save_generated_images(epoch, generator, latent_dim)