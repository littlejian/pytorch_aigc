import torch
device="cuda" if torch.cuda.is_available() else "cpu"

def onehot_encoder(position,depth):
    onehot=torch.zeros((depth,))
    onehot[position]=1
    return onehot

print(onehot_encoder(1,5))

def int_to_onehot(number):
    onehot=onehot_encoder(number,100)
    return onehot

onehot75=int_to_onehot(75)
print(onehot75)

def onehot_to_int(onehot):
    num=torch.argmax(onehot)
    return num.item()

print(onehot_to_int(onehot75))

def gen_sequence():
    indices = torch.randint(0, 20, (10,))
    values = indices*5
    return values  

sequence=gen_sequence()
print(sequence)

import numpy as np

def gen_batch():
    # 创建一个由 10 个数字组成的序列，所有数字都是 5 的倍数
    sequence=gen_sequence()
    # 将每个整数转换为一个 100 维的独热编码变量
    batch=[int_to_onehot(i).numpy() for i in sequence]
    batch=np.array(batch)
    return torch.tensor(batch)
batch=gen_batch()

def data_to_num(data):
    # 根据 100 维向量中的最大值，将向量转换为整数
    num=torch.argmax(data,dim=-1)
    return num
numbers=data_to_num(batch)

from torch import nn
D=nn.Sequential(
        nn.Linear(100,1),
        nn.Sigmoid()).to(device)

G=nn.Sequential(
        nn.Linear(100,100),
        nn.ReLU()).to(device)

loss_fn=nn.BCELoss()
lr=0.0005
optimD=torch.optim.Adam(D.parameters(),lr=lr)
optimG=torch.optim.Adam(G.parameters(),lr=lr)

real_labels=torch.ones((10,1)).to(device)
fake_labels=torch.zeros((10,1)).to(device)

def train_D_G(D,G,loss_fn,optimD,optimG):
    # 生成真实数据样本
    true_data=gen_batch().to(device)
    # 由于是真实样本，使用 1 作为标签
    preds=D(true_data)
    loss_D1=loss_fn(preds,real_labels.reshape(10,1))
    optimD.zero_grad()
    loss_D1.backward()
    optimD.step()
    # 在伪造数据上训练鉴别器
    noise=torch.randn(10,100).to(device)
    generated_data=G(noise)
    # 由于是伪造样本，使用 0 作为标签
    preds=D(generated_data)
    loss_D2=loss_fn(preds,fake_labels.reshape(10,1))
    optimD.zero_grad()
    loss_D2.backward()
    optimD.step()
    
    # 训练生成器 
    noise=torch.randn(10,100).to(device)
    generated_data=G(noise)
    # 使用 1 作为标签，因为生成器想要欺骗鉴别器
    preds=D(generated_data)
    loss_G=loss_fn(preds,real_labels.reshape(10,1))
    optimG.zero_grad()
    loss_G.backward()
    optimG.step()
    return generated_data

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

stopper=EarlyStop(800)    # 创建 Earlytop() 类实例

mse=nn.MSELoss()
real_labels=torch.ones((10,1)).to(device)
fake_labels=torch.zeros((10,1)).to(device)
# 定义 distance() 函数，用于计算生成数字的损失
def distance(generated_data): 
    nums=data_to_num(generated_data)
    remainders=nums%5
    ten_zeros=torch.zeros((10,1)).to(device)
    mseloss=mse(remainders,ten_zeros)
    return mseloss

for i in range(10000):
    gloss=0
    dloss=0
    generated_data=train_D_G(D,G,loss_fn,optimD,optimG)    # 训练 GAN 一个 epoch 
    dis=distance(generated_data)
    if stopper.stop(dis)==True:
        break   
    # 每训练 50 个 epoch 后，打印生成的整数序列
    if i % 50 == 0:
        print(data_to_num(generated_data))

import os
os.makedirs("files", exist_ok=True)
scripted = torch.jit.script(G) 
scripted.save('files/num_gen.pt') 

# 加载已保存的生成器
new_G=torch.jit.load('files/num_gen.pt',
                     map_location=device)
new_G.eval()
# 获取随机噪声向量
noise=torch.randn((10,100)).to(device)
# 将随机噪声向量输入训练好的模型，生成一系列整数
new_data=new_G(noise) 
print(data_to_num(new_data))