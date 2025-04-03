import pandas as pd
import os, shutil

# 加载包含图像属性的 CSV 文件
df=pd.read_csv("files/list_attr_celeba.csv")
# 创建两个文件夹用于存储黑发和金发的图像
os.makedirs("files/black", exist_ok=True)  
os.makedirs("files/blond", exist_ok=True) 
folder="files/img_align_celeba/img_align_celeba"
for i in range(len(df)):
    dfi=df.iloc[i]
    # 如果属性 Black_Hair 为 1，则将图像移动到 black 文件夹
    if dfi['Black_Hair']==1:
        try:
            oldpath=f"{folder}/{dfi['image_id']}"
            newpath=f"files/black/{dfi['image_id']}"
            shutil.move(oldpath, newpath)
        except:
            pass
    # 如果属性 Blond_Hair 为 1，则将图像移动到 blond 文件夹
    elif dfi['Blond_Hair']==1:
        try:
            oldpath=f"{folder}/{dfi['image_id']}"
            newpath=f"files/blond/{dfi['image_id']}"
            shutil.move(oldpath, newpath)
        except:
            pass

trainA=r"files/black/"
trainB=r"files/blond/"

import random
import matplotlib.pyplot as plt
from PIL import Image

imgs=os.listdir(trainA)
samples=random.sample(imgs,8)
imgs1=os.listdir(trainB)
samples1=random.sample(imgs1,8)
fs=[trainA,trainB]
ps=[imgs,imgs1]
fig=plt.figure(dpi=100, figsize=(1.78*8,2.18*2))
for i in range(16):
    ax = plt.subplot(2, 8, i + 1)
    folder=i//8
    p=i%8
    img=Image.open(fr"{fs[folder]}{ps[folder][p]}")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=-0.01,hspace=-0.1)
plt.show() 

from util import LoadData
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations 
from albumentations.pytorch import ToTensorV2

transforms = albumentations.Compose(
    [albumentations.Resize(width=256, height=256),    # 将图像大小调整为 256 x 256 像素
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],max_pixel_value=255),    # 将图像归一化到 [-1, 1] 范围内
        ToTensorV2()],
    additional_targets={"image0": "image"}) 
dataset = LoadData(root_A=["files/black/"],
    root_B=["files/blond/"],
    transform=transforms)    # 对图像应用 LoadData() 类
loader=DataLoader(dataset,batch_size=1,
    shuffle=True, pin_memory=True)    # 创建数据迭代器

from util import Discriminator, weights_init
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# 创建 Discriminator 类实例
disc_A = Discriminator().to(device)
disc_B = Discriminator().to(device)
# 初始化权重
weights_init(disc_A)
weights_init(disc_B)

from util import Generator

gen_A = Generator(img_channels=3, num_residuals=9).to(device)
gen_B = Generator(img_channels=3, num_residuals=9).to(device)
weights_init(gen_A)
weights_init(gen_B)

import torch.nn as nn

l1 = nn.L1Loss()
mse = nn.MSELoss()
g_scaler = torch.amp.GradScaler('cuda')
d_scaler = torch.amp.GradScaler('cuda')

lr = 0.00001
opt_disc = torch.optim.Adam(list(disc_A.parameters()) + 
  list(disc_B.parameters()),lr=lr,betas=(0.5, 0.999))
opt_gen = torch.optim.Adam(list(gen_A.parameters()) + 
  list(gen_B.parameters()),lr=lr,betas=(0.5, 0.999))

from util import train_epoch

for epoch in range(1):
    # 使用黑发和金发图像训练 CycleGAN 一个 epoch
    train_epoch(disc_A, disc_B, gen_A, gen_B, loader, opt_disc,
    opt_gen, l1, mse, d_scaler, g_scaler, device)
# 保存训练好的模型权重
torch.save(gen_A.state_dict(), "files/gen_black.pth")
torch.save(gen_B.state_dict(), "files/gen_blond.pth")

from torchvision.utils import save_image

gen_A.load_state_dict(torch.load("files/gen_black.pth"))
gen_B.load_state_dict(torch.load("files/gen_blond.pth"))
i=1
for black,blond in loader:
    fake_blond=gen_B(black.to(device))
    # 原始黑发图像
    save_image(black*0.5+0.5,f"files/black{i}.png")
    save_image(fake_blond*0.5+0.5,f"files/fakeblond{i}.png")   
    fake2black=gen_A(fake_blond)
    # 经过转换后的黑发生成图像
    save_image(fake2black*0.5+0.5,f"files/fake2black{i}.png")    
    fake_black=gen_A(blond.to(device))
    # 原始金发图像
    save_image(blond*0.5+0.5,f"files/blond{i}.png")
    save_image(fake_black*0.5+0.5,f"files/fakeblack{i}.png")
    fake2blond=gen_B(fake_black)
    # 经过转换后的金发生成图像
    save_image(fake2blond*0.5+0.5,f"files/fake2blond{i}.png")  
    i=i+1
    if i>10:
        break