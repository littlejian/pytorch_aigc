import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

transform=T.Compose([
    T.ToTensor(),
    T.Normalize([0.5],[0.5])])

train_set=torchvision.datasets.FashionMNIST(
        root=".",
        train=True,
        download=True,
        transform=transform)

batch_size=32
train_loader=torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True)

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

images, labels = next(iter(train_loader))
# Plot all the images of the 1st batch in grid
grid = make_grid(0.5-images/2, 8, 4)
plt.imshow(grid.numpy().transpose((1, 2, 0)),
           cmap="gray_r")
plt.axis("off")
plt.show()

import torch.nn as nn

device="cuda" if torch.cuda.is_available() else "cpu"

D=nn.Sequential(
        nn.Linear(784, 1024), # 第一个全连接层有 784 个输入和 1,024 个输出
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),  # 最后一个全连接层有 256 个输入和 1 个输出
        nn.Sigmoid()).to(device)

G=nn.Sequential(
        nn.Linear(100, 256),  # 生成器中的第一个层与鉴别器中的最后一层对称
        nn.ReLU(),
        nn.Linear(256, 512),  # 生成器中的第二个层与鉴别器中的倒数第二层对称
        nn.ReLU(),
        nn.Linear(512, 1024), # 生成器中的第三个层与鉴别器中的倒数第三层对称
        nn.ReLU(),
        nn.Linear(1024, 784), # 生成器中的最后一层与鉴别器中的第一层对称
        nn.Tanh()).to(device) # 使用 Tanh() 激活函数，使得输出值在 -1 和 1 之间，与图像中的值相同

loss_fn=nn.BCELoss()
lr=0.0001
optimD=torch.optim.Adam(D.parameters(),lr=lr)
optimG=torch.optim.Adam(G.parameters(),lr=lr)

import matplotlib.pyplot as plt

def see_output():
    noise=torch.randn(32,100).to(device=device)
    fake_samples=G(noise).cpu().detach()    # 生成 32 张伪造图像
    plt.figure(dpi=100,figsize=(20,10))
    for i in range(32):
        ax=plt.subplot(4, 8, i + 1)
        img=(fake_samples[i]/2+0.5).reshape(28, 28)
        plt.imshow(img)    # 图像可视化
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
see_output()

real_labels=torch.ones((batch_size,1)).to(device)
fake_labels=torch.zeros((batch_size,1)).to(device)

def train_D_on_real(real_samples):
    #r=torch.FloatTensor(real_samples)
    r=real_samples.reshape(-1,28*28).to(device)
    out_D=D(r)    
    labels=torch.ones((r.shape[0],1)).to(device)
    loss_D=loss_fn(out_D,labels)    
    optimD.zero_grad()
    loss_D.backward()
    optimD.step()    
    return loss_D

def train_D_on_fake():        
    noise=torch.randn(batch_size,100).to(device=device)
    generated_data=G(noise)
    preds=D(generated_data)
    loss_D=loss_fn(preds,fake_labels)
    optimD.zero_grad()
    loss_D.backward()
    optimD.step()
    return loss_D

def train_G(): 
    noise=torch.randn(batch_size,100).to(device=device)
    generated_data=G(noise)
    preds=D(generated_data)
    loss_G=loss_fn(preds,real_labels)
    optimG.zero_grad()
    loss_G.backward()
    optimG.step()
    return loss_G

for i in range(50):
    gloss=0
    dloss=0
    for n, (real_samples,_) in enumerate(train_loader):
        loss_D=train_D_on_real(real_samples) # 使用真实样本训练鉴别器
        dloss+=loss_D
        loss_D=train_D_on_fake()    # 使用伪造样本训练鉴别器
        dloss+=loss_D
        loss_G=train_G()            # 训练生成器
        gloss+=loss_G
    gloss=gloss/n
    dloss=dloss/n
    # 每隔 10 个 epoch 可视化生成图像
    if i % 10 == 9:
        print(f"at epoch {i+1}, dloss: {dloss}, gloss {gloss}")
        see_output()

import os
scripted = torch.jit.script(G) 
os.makedirs("files", exist_ok=True)
scripted.save('files/fashion_gen.pt') 

new_G=torch.jit.load('files/fashion_gen.pt',
                     map_location=device)
new_G.eval()

noise=torch.randn(batch_size,100).to(device=device)
fake_samples=new_G(noise).cpu().detach()
for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow((fake_samples[i]/2+0.5).reshape(28, 28))
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(hspace=-0.6)
plt.show()