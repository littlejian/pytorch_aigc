import pandas as pd

# 加载文件 train.csv
train=pd.read_csv('files/train.csv')
# 将 id 列中的值设置为数据索引
train.set_index('id', inplace=True)

print(train["glasses"])

import os, shutil

G='files/glasses/G/'
NoG='files/glasses/NoG/'
os.makedirs(G, exist_ok=True)    # 创建子文件夹 files/glasses/G/ 用于存放带眼镜的图像
os.makedirs(NoG, exist_ok=True)    # 创建子文件夹 files/glasses/NoG/ 用于存放不带眼镜的图像
folder='files/faces-spring-2020/faces-spring-2020/'
for i in range(1,4501):
    oldpath=f"{folder}face-{i}.png"
    if train.loc[i]['glasses']==0:    # 将标记为 0 的图像移动到 NoG 文件夹
        newpath=f"{NoG}face-{i}.png"
    elif train.loc[i]['glasses']==1:    # 将标记为 1 的图像移动到 G 文件夹
        newpath=f"{G}face-{i}.png"
    shutil.move(oldpath, newpath)

import random
import matplotlib.pyplot as plt
from PIL import Image

imgs=os.listdir(G)

# 从文件夹 G 中随机选择 16 张图像
samples=random.sample(imgs,16)
fig=plt.figure(dpi=100, figsize=(8,2))
for i in range(16):
    ax = plt.subplot(2, 8, i + 1)
    img=Image.open(f"{G}{samples[i]}")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=-0.01,hspace=-0.01)
plt.show()

imgs=os.listdir(NoG)
samples=random.sample(imgs,16)
fig=plt.figure(dpi=100, figsize=(8,2))
for i in range(16):
    ax = plt.subplot(2, 8, i + 1)
    img=Image.open(f"{NoG}{samples[i]}")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=-0.01,hspace=-0.01)
plt.show()

import torch.nn as nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
class Critic(nn.Module):
    def __init__(self, img_channels, features):
        super().__init__()
        # 评论家网络有两个 Conv2d 层和五个块
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, features, 
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self.block(features, features * 2, 4, 2, 1),
            self.block(features * 2, features * 4, 4, 2, 1),
            self.block(features * 4, features * 8, 4, 2, 1),
            self.block(features * 8, features * 16, 4, 2, 1),  
            self.block(features * 16, features * 32, 4, 2, 1),            
            nn.Conv2d(features * 32, 1, kernel_size=4,
                      stride=2, padding=0)) # 输出包含一个特征值
    # 每个块包含一个 Conv2d 层、一个 InstanceNorm2d 层，并使用 LeakyReLU 激活函数
    def block(self, in_channels, out_channels, 
              kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,
                kernel_size,stride,padding,bias=False,),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2))
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, noise_channels, img_channels, features):
        super(Generator, self).__init__()
        # 生成器由七个 ConvTranspose2d 层组成
        self.net = nn.Sequential(
            self.block(noise_channels, features *64, 4, 1, 0),
            self.block(features * 64, features * 32, 4, 2, 1),
            self.block(features * 32, features * 16, 4, 2, 1),
            self.block(features * 16, features * 8, 4, 2, 1),
            self.block(features * 8, features * 4, 4, 2, 1),            
            self.block(features * 4, features * 2, 4, 2, 1),            
            nn.ConvTranspose2d(
                features * 2, img_channels, kernel_size=4,
                stride=2, padding=1),
            nn.Tanh()) # 使用 Tanh 激活函数将值压缩到 [-1, 1] 范围内，和训练集中的图像相同
    # 每个块由一个 ConvTranspose2d 层、一个 BatchNorm2d 层和 ReLU 激活函数组成
    def block(self, in_channels, out_channels, 
              kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,
                kernel_size,stride,padding,bias=False,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
    def forward(self, x):
        return self.net(x)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)   

z_dim=100
img_channels=3
features=16
gen=Generator(z_dim+2,img_channels,features).to(device)
critic=Critic(img_channels+2,features).to(device)
weights_init(gen)
weights_init(critic)

lr = 0.0001
opt_gen = torch.optim.Adam(gen.parameters(), 
                         lr = lr, betas=(0.0, 0.9))
opt_critic = torch.optim.Adam(critic.parameters(), 
                         lr = lr, betas=(0.0, 0.9))

def GP(critic, real, fake):
    B, C, H, W = real.shape
    alpha=torch.rand((B,1,1,1)).repeat(1,C,H,W).to(device)
    # 创建真实图像和伪造图像的插值图像
    interpolated_images = real*alpha+fake*(1-alpha)
    # 获取评论家对于插值图像的评分
    critic_scores = critic(interpolated_images)
    # 计算梯度
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=critic_scores,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True)[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    # 梯度惩罚是梯度范数与值 1 之差的平方
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

import torchvision.transforms as T
import torchvision

batch_size=16
imgsz=256
transform=T.Compose([
    T.Resize((imgsz,imgsz)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])      
data_set=torchvision.datasets.ImageFolder(
    root=r"files/glasses",
    transform=transform) 

newdata=[]
for i,(img,label) in enumerate(data_set):
    onehot=torch.zeros((2))
    onehot[label]=1
    # 创建两个额外的通道，填充为 0，每个通道的形状为 256 x 256，与输入图像中每个通道的维度相同
    channels=torch.zeros((2,imgsz,imgsz))
    if label==0:
        # 如果原始图像标签为 0，则将第四个通道填充为 1
        channels[0,:,:]=1
    else:
        # 如果原始图像标签为 4，则将第五个通道填充为 1
        channels[1,:,:]=1
    # 将第四个和第五个通道添加到原始图像中，形成一个五通道的标签图像
    img_and_label=torch.cat([img,channels],dim=0)    
    newdata.append((img,label,onehot,img_and_label))

data_loader=torch.utils.data.DataLoader(
    newdata,batch_size=batch_size,shuffle=True)

def plot_epoch(epoch):
    noise = torch.randn(32, z_dim, 1, 1)
    labels = torch.zeros(32, 2, 1, 1)
    # 为带眼镜图片创建独热编码标签
    labels[:,0,:,:]=1
    noise_and_labels=torch.cat([noise,labels],dim=1).to(device)
    # 将拼接的噪声向量和标签输入生成器，以生成带眼镜的图像
    fake=gen(noise_and_labels).cpu().detach()
    fig=plt.figure(figsize=(20,10),dpi=100)
    # 绘制生成的带眼镜的图像
    for i in range(32):
        ax = plt.subplot(4, 8, i + 1)
        img=(fake.cpu().detach()[i]/2+0.5).permute(1,2,0)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(hspace=-0.6)
    plt.savefig(f"files/glasses/G{epoch}.png")
    plt.show() 
    noise = torch.randn(32, z_dim, 1, 1)
    labels = torch.zeros(32, 2, 1, 1)
    # 为不带眼镜的图像创建独热编码标签
    labels[:,1,:,:]=1
    noise_and_labels=torch.cat([noise,labels],dim=1).to(device)
    fake=gen(noise_and_labels).cpu().detach()
    fig=plt.figure(figsize=(20,10),dpi=100)
    for i in range(32):
        ax = plt.subplot(4, 8, i + 1)
        img=(fake.cpu().detach()[i]/2+0.5).permute(1,2,0)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(hspace=-0.6)
    plt.savefig(f"files/glasses/NoG{epoch}.png")
    plt.show()

def train_batch(onehots,img_and_labels,epoch):
    # 一批带标签的真实图像
    real = img_and_labels.to(device)
    B = real.shape[0]
    for _ in range(5):
        noise = torch.randn(B, z_dim, 1, 1)
        onehots=onehots.reshape(B,2,1,1)
        noise_and_labels=torch.cat([noise,onehots],dim=1).to(device)
        fake_img = gen(noise_and_labels).to(device)
        fakelabels=img_and_labels[:,3:,:,:].to(device)
        # 一批带标签的生成图像
        fake=torch.cat([fake_img,fakelabels],dim=1).to(device)
        critic_real = critic(real).reshape(-1)
        critic_fake = critic(fake).reshape(-1)
        gp = GP(critic, real, fake)
        # 评论家的总损失由三个部分组成：评估真实图像的损失、评估伪造图像的损失和梯度惩罚损失
        loss_critic=(-(torch.mean(critic_real) - 
           torch.mean(critic_fake)) + 10 * gp)
        opt_critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()
    gen_fake = critic(fake).reshape(-1)
    # 训练生成器
    loss_gen = -torch.mean(gen_fake)
    opt_gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()
    return loss_critic, loss_gen

for epoch in range(1,101):
    closs=0
    gloss=0
    # 遍历训练数据集中的所有批次
    for _,_,onehots,img_and_labels in data_loader:    
        # 使用一批数据训练模型
        loss_critic, loss_gen = train_batch(onehots, img_and_labels,epoch)   
        closs+=loss_critic.detach()/len(data_loader)
        gloss+=loss_gen.detach()/len(data_loader)
    print(f"at epoch {epoch}, critic loss: {closs}, generator loss {gloss}")
    plot_epoch(epoch)

# 保存训练好的生成器权重
torch.save(gen.state_dict(),'files/cgan.pth')

generator=Generator(z_dim+2,img_channels,features).to(device)
# 加载训练好的权重
generator.load_state_dict(torch.load("files/cgan.pth", map_location=device))
generator.eval()

# 生成一组随机噪声向量并保存，以便可以从中选择特定的向量来执行向量运算
noise_g = torch.randn(32, z_dim, 1, 1)
labels_g = torch.zeros(32, 2, 1, 1)
# 创建一个标签，用于生成带眼镜的图像
labels_g[:,0,:,:]=1
noise_and_labels=torch.cat([noise_g,labels_g],dim=1).to(device)
fake=generator(noise_and_labels)
plt.figure(figsize=(20,10),dpi=50)
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    img=(fake.cpu().detach()[i]/2+0.5).permute(1,2,0)
    plt.imshow(img.numpy())
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=-0.08,hspace=-0.01)
plt.show()

z_male_g=noise_g[0]
z_female_g=noise_g[14]

noise_ng = torch.randn(32, z_dim, 1, 1)
labels_ng = torch.zeros(32, 2, 1, 1)
labels_ng[:,1,:,:]=1
noise_and_labels=torch.cat([noise_ng,labels_ng],dim=1).to(device)
fake=generator(noise_and_labels).cpu().detach()
plt.figure(figsize=(20,10),dpi=50)
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    img=(fake.cpu().detach()[i]/2+0.5).permute(1,2,0)
    plt.imshow(img.numpy())#.repeat(4,axis=0).repeat(4,axis=1))
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=-0.08,hspace=-0.01)
plt.show()

z_male_ng=noise_ng[8]
z_female_ng=noise_ng[31]

# 创建 5 个权重
weights=[0,0.25,0.5,0.75,1]
plt.figure(figsize=(20,4),dpi=50)
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    # 创建两个标签的加权平均值
    label=weights[i]*labels_ng[0]+(1-weights[i])*labels_g[0]
    noise_and_labels=torch.cat(
        [z_female_g.reshape(1, z_dim, 1, 1),
         label.reshape(1, 2, 1, 1)],dim=1).to(device)
    # 将新的标签输入训练好的模型以生成图像
    fake=generator(noise_and_labels).cpu().detach()    
    img=(fake[0]/2+0.5).permute(1,2,0)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=-0.08,hspace=-0.01)
plt.show()

# 创建 5 个权重
weights=[0,0.25,0.5,0.75,1]
plt.figure(figsize=(20,4),dpi=50)
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    # 创建两个随机噪声向量的加权平均值
    z=weights[i]*z_female_ng+(1-weights[i])*z_male_ng
    noise_and_labels=torch.cat(
        [z.reshape(1, z_dim, 1, 1),
         labels_ng[0].reshape(1, 2, 1, 1)],dim=1).to(device)
    # 将新的随机噪声向量输入训练好的模型以生成图像  
    fake=generator(noise_and_labels).cpu().detach()    
    img=(fake[0]/2+0.5).permute(1,2,0)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=-0.08,hspace=-0.01)
plt.show()

plt.figure(figsize=(20,5),dpi=50)
for i in range(4):
    ax = plt.subplot(1, 4, i + 1)
    p=i//2
    q=i%2 
    # p 的值，可以是 0 或 1，用于选择随机噪声向量，以生成男性或女性面孔
    z=z_female_g*p+z_male_g*(1-p)
    # q 的值，可以是 0 或 1，用于选择标签，以确定生成的图像是否带有眼镜
    label=labels_ng[0]*q+labels_g[0]*(1-q)
    # 将随机噪声向量与标签结合，以选择两个特征
    noise_and_labels=torch.cat(
        [z.reshape(1, z_dim, 1, 1),
         label.reshape(1, 2, 1, 1)],dim=1).to(device)
    fake=generator(noise_and_labels)
    img=(fake.cpu().detach()[0]/2+0.5).permute(1,2,0)
    plt.imshow(img.numpy())
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=-0.08,hspace=-0.01)
plt.show()

plt.figure(figsize=(20,20),dpi=50)
for i in range(36):
    ax = plt.subplot(6,6, i + 1)
    p=i//6
    q=i%6 
    z=z_female_ng*p/5+z_male_ng*(1-p/5)
    label=labels_ng[0]*q/5+labels_g[0]*(1-q/5)
    noise_and_labels=torch.cat(
        [z.reshape(1, z_dim, 1, 1),
         label.reshape(1, 2, 1, 1)],dim=1).to(device)
    fake=generator(noise_and_labels)
    img=(fake.cpu().detach()[0]/2+0.5).permute(1,2,0)
    plt.imshow(img.numpy())
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=-0.08,hspace=-0.01)
plt.show()