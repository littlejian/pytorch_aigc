import torch

observations = 2048
# 创建一个具有 2,048 行和 2 列的张量
train_data = torch.zeros((observations, 2))
# 生成介于 0 和 50 之间的 x 值
train_data[:,0]=50*torch.rand(observations)
# 根据关系 y = 1.08^x 生成 y 值
train_data[:,1]=1.08**train_data[:,0]

import matplotlib.pyplot as plt

fig=plt.figure(dpi=100,figsize=(8,6))
# 绘制 x 和 y 之间的关系
plt.plot(train_data[:,0],train_data[:,1],".",c="r")
plt.xlabel("values of x",fontsize=15)
# 标记 y 轴
plt.ylabel("values of $y=1.08^x$",fontsize=15)
# 创建标题
plt.title("An exponential growth shape",fontsize=20)
plt.show()

from torch.utils.data import DataLoader

batch_size=128
train_loader=DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True)

batch0=next(iter(train_loader))
print(batch0)

import torch.nn as nn

device="cuda" if torch.cuda.is_available() else "cpu"

D=nn.Sequential(
        # 第一层的输入特征数量是 2，与每个数据实例中的元素数量匹配，每个数据样本有两个值，x 和 y
        nn.Linear(2,256),
        nn.ReLU(),
        # Dropout 层防止过拟合
        nn.Dropout(0.3),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Dropout(0.3),
        # 最后一层的输出特征数量是 1，将其压缩为一个介于 0 和 1 之间的值
        nn.Linear(64,1),
        nn.Sigmoid()).to(device)

G=nn.Sequential(
        # 第一层的输入特征数量是 2，和潜空间中随机噪声向量的维度相同
        nn.Linear(2,16),
        nn.ReLU(),
        nn.Linear(16,32),
        nn.ReLU(),
        # 最后一层的输出特征数量是 2，和数据样本的维度相同，数据样本包含两个值 (x, y)
        nn.Linear(32,2)).to(device)

loss_fn=nn.BCELoss()
lr=0.0005
optimD=torch.optim.Adam(D.parameters(),lr=lr)
optimG=torch.optim.Adam(G.parameters(),lr=lr)

# 使用均方误差 (MSE) 作为衡量性能的标准
mse=nn.MSELoss()

def performance(fake_samples):
    # 真实分布
    real=1.08**fake_samples[:,0]
    # 将生成分布与真实分布进行比较，并计算 MSE
    mseloss=mse(fake_samples[:,1],real)
    return mseloss

class EarlyStop:
    def __init__(self, patience=1000): # 将 patience 的默认值设置为 1000
        self.patience = patience
        self.steps = 0
        self.min_gdif = float('inf')
    def stop(self, gdif):    # 定义 stop() 方法
        # # 如果生成分布与真实分布之间的新差异大于当前最小差异，则更新 min_gdif 的值
        if gdif < self.min_gdif:
            self.min_gdif = gdif
            self.steps = 0
        elif gdif >= self.min_gdif:
            self.steps += 1
        # 如果模型在 1000 个 epoch 内没有改进，则停止训练
        if self.steps >= self.patience:
            return True
        else:
            return False

stopper=EarlyStop()

real_labels=torch.ones((batch_size,1))
real_labels=real_labels.to(device)

fake_labels=torch.zeros((batch_size,1))
fake_labels=fake_labels.to(device)

def train_D_on_real(real_samples):
    real_samples=real_samples.to(device)
    optimD.zero_grad()
    out_D=D(real_samples)   # 对真实样本进行预测
    loss_D=loss_fn(out_D,real_labels)    # 计算损失
    loss_D.backward()
    optimD.step()    # 反向传播
    return loss_D

def train_D_on_fake():        
    noise=torch.randn((batch_size,2))
    noise=noise.to(device)
    fake_samples=G(noise)     # 生成一批伪造样本      
    optimD.zero_grad()
    out_D=D(fake_samples)    # 对伪造样本进行预测
    loss_D=loss_fn(out_D,fake_labels)    # 计算损失
    loss_D.backward()
    optimD.step()    # 反向传播
    return loss_D

def train_G(): 
    noise=torch.randn((batch_size,2))
    noise=noise.to(device)
    optimG.zero_grad()
    fake_samples=G(noise)    # 生成一批伪造样本
    out_G=D(fake_samples)    # 将伪造样本输入到鉴别器以获得预测
    loss_G=loss_fn(out_G,real_labels)    # 计算损失
    loss_G.backward()
    optimG.step()     # 反向传播
    return loss_G, fake_samples

import os
os.makedirs("files", exist_ok=True)    # 创建文件夹用于保存生成结果

def test_epoch(epoch,gloss,dloss,n,fake_samples):
    if epoch==0 or (epoch+1)%25==0:
        g=gloss.item()/n
        d=dloss.item()/n
        print(f"at epoch {epoch+1}, G loss: {g}, D loss {d}")    # 定期打印损失值
        fake=fake_samples.detach().cpu().numpy()
        plt.figure(dpi=200)
        plt.plot(fake[:,0],fake[:,1],"*",c="g",
            label="generated samples")    # 将生成数据点绘制为 *
        plt.plot(train_data[:,0],train_data[:,1],".",c="r",
            alpha=0.1,label="real samples")    # 将训练数据绘制为 .
        plt.title(f"epoch {epoch+1}")
        plt.xlim(0,50)
        plt.ylim(0,50)
        plt.legend()
        plt.savefig(f"files/p{epoch+1}.png")

# 开始训练循环
for epoch in range(10000):
    gloss=0
    dloss=0
    for n, real_samples in enumerate(train_loader):    # 遍历训练数据集中的所有批次
        loss_D=train_D_on_real(real_samples)
        dloss+=loss_D
        loss_D=train_D_on_fake()
        dloss+=loss_D
        loss_G,fake_samples=train_G()
        gloss+=loss_G
    test_epoch(epoch,gloss,dloss,n,fake_samples)    # 定期显示生成的样本
    gdif=performance(fake_samples).item()
    if stopper.stop(gdif)==True:    #D 判断是否应该停止训练
        break

import os
os.makedirs("files", exist_ok=True)
scripted = torch.jit.script(G) 
scripted.save('files/exponential.pt') 

new_G=torch.jit.load('files/exponential.pt',
                     map_location=device)
new_G.eval()

noise=torch.randn((batch_size,2)).to(device)
new_data=new_G(noise) 

fig=plt.figure(dpi=100)
# 将生成的数据样本绘制为 *
plt.plot(new_data.detach().cpu().numpy()[:,0],
        new_data.detach().cpu().numpy()[:,1],"*",c="g",
        label="generated samples")
# 将训练数据绘制为 .
plt.plot(train_data[:,0],train_data[:,1],".",c="r",
         alpha=0.1,label="real samples")
plt.title("Inverted-U Shape Generated by GANs")
plt.xlim(0,50)
plt.ylim(0,50)
plt.legend()
plt.show()