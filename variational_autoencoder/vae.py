import torchvision
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from torch import nn

device="cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
            T.Resize(256), # 将图像大小调整为 256 × 256 像素
            T.ToTensor(),  # 将图像转换为值在 0 和 1 之间的张量
            ])
# 加载图像并应用转换
data = torchvision.datasets.ImageFolder(
    root="files/glasses",
    transform=transform)
batch_size=16
# 创建数据加载器
loader = torch.utils.data.DataLoader(data,
     batch_size=batch_size,shuffle=True)

# 潜空间的维度为 100
latent_dims=100
class Encoder(nn.Module):
    def __init__(self, latent_dims=100):  
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)  
        self.linear1 = nn.Linear(31*31*32, 1024)
        self.linear2 = nn.Linear(1024, latent_dims)
        self.linear3 = nn.Linear(1024, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() 
        self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        # 编码分布的均值
        mu =  self.linear2(x)
        # 编码分布的标准差
        std = torch.exp(self.linear3(x))
        # 编码向量表示
        z = mu + std*self.N.sample(mu.shape)
        return mu, std, z

class Decoder(nn.Module):   
    def __init__(self, latent_dims=100):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 1024), # 潜编码首先通过两个全连接层
            nn.ReLU(True),
            nn.Linear(1024, 31*31*32),
            nn.ReLU(True))
        # 将潜编码重塑为多维对象
        self.unflatten = nn.Unflatten(dim=1, 
                  unflattened_size=(32,31,31))
        # 潜编码通过三个转置卷积层
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32,16,3,stride=2,
                               output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2,
                               padding=1, output_padding=1))
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        # 将输出压缩为 0 和 1 之间
        x = torch.sigmoid(x)
        return x
    
class VAE(nn.Module):
    def __init__(self, latent_dims=100):
        super().__init__()
        # 通过实例化 Encoder() 类创建编码器
        self.encoder = Encoder(latent_dims)
        # 通过实例化 Decoder() 类创建解码器
        self.decoder = Decoder(latent_dims)
    def forward(self, x):
        x = x.to(device)
        # 将输入通过编码器传递以获得潜编码
        mu, std, z = self.encoder(x)
        # 返回潜编码的均值和标准差，以及重建的图像
        return mu, std, self.decoder(z) 
    
vae=VAE().to(device)
lr=1e-4 
optimizer=torch.optim.Adam(vae.parameters(),
                           lr=lr,weight_decay=1e-5)

def train_epoch(epoch):
    vae.train()
    epoch_loss = 0.0
    for imgs, _ in loader: 
        imgs = imgs.to(device)
        mu, std, out = vae(imgs)
        # 计算重建损失
        reconstruction_loss = ((imgs-out)**2).sum() 
        # 计算 KL 损失
        kl = ((std**2)/2 + (mu**2)/2 - torch.log(std) - 0.5).sum()
        # 总损失
        loss = reconstruction_loss + kl
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    print(f'at epoch {epoch}, loss is {epoch_loss}')

import numpy as np
import matplotlib.pyplot as plt

def plot_epoch():
    with torch.no_grad():
        noise = torch.randn(18,latent_dims).to(device)
        imgs = vae.decoder(noise).cpu()
        imgs = torchvision.utils.make_grid(imgs,6,3).numpy()
        fig, ax = plt.subplots(figsize=(6,3),dpi=100)
        plt.imshow(np.transpose(imgs, (1, 2, 0)))
        plt.axis("off")
        plt.show()

for epoch in range(1,11):
    train_epoch(epoch)
    plot_epoch()
torch.save(vae.state_dict(),"files/VAEglasses.pth")    

vae.eval()
vae.load_state_dict(torch.load('files/VAEglasses.pth',
                              map_location=device))

imgs,_=next(iter(loader))
imgs = imgs.to(device)
mu, std, out = vae(imgs)
images=torch.cat([imgs,out],dim=0).detach().cpu()
images = torchvision.utils.make_grid(images,8,4)
fig, ax = plt.subplots(figsize=(8,4),dpi=100)
plt.imshow(np.transpose(images, (1, 2, 0)))
plt.axis("off")
plt.show()

plot_epoch()

glasses=[]
for i in range(25):
    img,label=data[i]
    glasses.append(img)
    plt.subplot(5,5,i+1)
    plt.imshow(img.numpy().transpose((1,2,0)))
    plt.axis("off")
plt.show()
# 选择三张带眼镜的男性图像  
men_g=[glasses[0],glasses[3],glasses[14]]
# 选择三张带眼镜的女性图像
women_g=[glasses[9],glasses[15],glasses[21]] 

noglasses=[]
for i in range(25):
    img,label=data[-i-1]
    noglasses.append(img)
    plt.subplot(5,5,i+1)
    plt.imshow(img.numpy().transpose((1,2,0)))
    plt.axis("off")
plt.show()
# 选择三张不带眼镜的男性图像
men_ng=[noglasses[1],noglasses[7],noglasses[22]]
# 选择三张不带眼镜的女性图像
women_ng=[noglasses[4],noglasses[9],noglasses[19]]

# 创建一批戴眼镜的男性图像
men_g_batch = torch.cat((men_g[0].unsqueeze(0),
             men_g[1].unsqueeze(0),
             men_g[2].unsqueeze(0)), dim=0).to(device)
_,_,men_g_encodings=vae.encoder(men_g_batch)
# 获取戴眼镜男性的平均潜编码
men_g_encoding=men_g_encodings.mean(dim=0)
# 解码戴眼镜男性的平均潜编码
men_g_recon=vae.decoder(men_g_encoding.unsqueeze(0))

women_g_batch = torch.cat((women_g[0].unsqueeze(0),
             women_g[1].unsqueeze(0),
             women_g[2].unsqueeze(0)), dim=0).to(device)
men_ng_batch = torch.cat((men_ng[0].unsqueeze(0),
             men_ng[1].unsqueeze(0),
             men_ng[2].unsqueeze(0)), dim=0).to(device)
women_ng_batch = torch.cat((women_ng[0].unsqueeze(0),
             women_ng[1].unsqueeze(0),
             women_ng[2].unsqueeze(0)), dim=0).to(device)
# 获取其他三组的平均潜编码
_,_,women_g_encodings=vae.encoder(women_g_batch)
women_g_encoding=women_g_encodings.mean(dim=0)
_,_,men_ng_encodings=vae.encoder(men_ng_batch)
men_ng_encoding=men_ng_encodings.mean(dim=0)
_,_,women_ng_encodings=vae.encoder(women_ng_batch)
women_ng_encoding=women_ng_encodings.mean(dim=0)
# 解码其他三组的平均潜编码
women_g_recon=vae.decoder(women_g_encoding.unsqueeze(0))
men_ng_recon=vae.decoder(men_ng_encoding.unsqueeze(0))
women_ng_recon=vae.decoder(women_ng_encoding.unsqueeze(0))

imgs=torch.cat((men_g_recon,
                women_g_recon,
                men_ng_recon,
                women_ng_recon),dim=0)
imgs=torchvision.utils.make_grid(imgs,4,1).cpu().numpy()
imgs=np.transpose(imgs,(1,2,0))
fig, ax = plt.subplots(figsize=(8,2),dpi=100)
plt.imshow(imgs)
plt.axis("off")
plt.show()

# 将 z 定义为戴眼镜的男性潜编码 + 不戴眼镜的女性潜编码
z=men_g_encoding-women_g_encoding+women_ng_encoding
# 解码 z 以生成图像
out=vae.decoder(z.unsqueeze(0))
imgs=torch.cat((men_g_recon,
                women_g_recon,
                women_ng_recon,out),dim=0)
imgs=torchvision.utils.make_grid(imgs,4,1).cpu().numpy()
imgs=np.transpose(imgs,(1,2,0))
fig, ax = plt.subplots(figsize=(8,2),dpi=100)
# 可视化图像
plt.imshow(imgs)
plt.title("man with glasses - woman \
with glasses + woman without \
glasses = man without glasses ",fontsize=10,c="r")
plt.axis("off")
plt.show()

results=[]
for w in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    z=w*women_ng_encoding+(1-w)*women_g_encoding
    out=vae.decoder(z.unsqueeze(0))
    results.append(out)
imgs=torch.cat((results[0],results[1],results[2],
                results[3],results[4],results[5]),dim=0)
imgs=torchvision.utils.make_grid(imgs,6,1).cpu().numpy()
imgs=np.transpose(imgs,(1,2,0))
fig, ax = plt.subplots(dpi=100)
plt.imshow(imgs)
plt.axis("off")
plt.show() 