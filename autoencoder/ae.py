import torchvision
import torchvision.transforms as T

transform=T.Compose([
    T.ToTensor()])
# 使用 torchvision.datasets 中的 MNIST() 类下载手写数字
train_set=torchvision.datasets.MNIST(root=".",
    train=True,download=True,transform=transform) 
test_set=torchvision.datasets.MNIST(root=".",
    train=False,download=True,transform=transform) 

import torch

batch_size=32
train_loader=torch.utils.data.DataLoader(
    train_set,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(
    test_set,batch_size=batch_size,shuffle=True)

import torch.nn.functional as F
from torch import nn

device="cuda" if torch.cuda.is_available() else "cpu"
# A输入包含 28 × 28 = 784 个值
input_dim = 784
# 潜变量（编码）包含 20 个值
z_dim = 20
h_dim = 200
class AE(nn.Module):
    def __init__(self,input_dim,z_dim,h_dim):
        super().__init__()
        self.common = nn.Linear(input_dim, h_dim)
        self.encoded = nn.Linear(h_dim, z_dim)
        self.l1 = nn.Linear(z_dim, h_dim)
        self.decode = nn.Linear(h_dim, input_dim)                
    def encoder(self, x):
        # 编码器将图像压缩为潜变量
        common = F.relu(self.common(x))
        mu = self.encoded(common)
        return mu
    def decoder(self, z):
        # 解码器根据编码重建图像
        out=F.relu(self.l1(z))
        out=torch.sigmoid(self.decode(out))
        return out
    def forward(self, x):
        mu=self.encoder(x)
        out=self.decoder(mu)
        return out, mu

# 编码器和解码器构成了 AE
model = AE(input_dim,z_dim,h_dim).to(device)
lr=0.00025
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

import matplotlib.pyplot as plt

# 收集测试集中每个数字的图像样本
originals = []
idx = 0
for img,label in test_set:
    if label == idx:
        originals.append(img)
        idx += 1
    if idx == 10:
        break

def plot_digits():
    reconstructed=[]
    for idx in range(10):
        with torch.no_grad():
            img = originals[idx].reshape((1,input_dim))
            # 将图像输入 AE 以获得重建图像
            out,mu = model(img.to(device))
        # 收集每个原始图像的重建图像
        reconstructed.append(out)
    # 比较原始图像与重建图像
    imgs=originals+reconstructed
    plt.figure(figsize=(10,2),dpi=50)
    for i in range(20):
        ax = plt.subplot(2,10, i + 1)
        img=(imgs[i]).detach().cpu().numpy()
        plt.imshow(img.reshape(28,28),
                   cmap="binary")
        plt.xticks([])
        plt.yticks([])
    plt.show()

plot_digits()

for epoch in range(50):
    tloss=0
    for imgs, labels in train_loader:
        # 重建图像
        imgs=imgs.to(device).view(-1, input_dim)
        out, mu=model(imgs)
        # 重建损失
        loss=((out-imgs)**2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tloss+=loss.item()
    print(f"at epoch {epoch} toal loss = {tloss}")
    # 可视化重建图像
    plot_digits()

scripted = torch.jit.script(model) 
scripted.save('files/AEdigits.pt') 

model=torch.jit.load('files/AEdigits.pt',map_location=device)
model.eval()

plot_digits()