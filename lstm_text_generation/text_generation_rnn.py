text="It is unbelievably good!"
tokens=list(text)
print(tokens)

text="It is unbelievably good!"
text=text.replace("!"," !")
tokens=text.split(" ")
print(tokens)

with open("files/anna.txt","r") as f:
    text=f.read()
words=text.split(" ") 
print(words[:20])

print(set(text.lower()))

# 将换行符替换为一个空格
clean_text=text.lower().replace("\n", " ")
# 将连字符替换为一个空格
clean_text=clean_text.replace("-", " ")
# 在标点符号和特殊字符周围添加空格
for x in ",.:;?!$()/_&%*@'`":
    clean_text=clean_text.replace(f"{x}", f" {x} ")
clean_text=clean_text.replace('"', ' " ') 
text=clean_text.split()

from collections import Counter   
word_counts = Counter(text)    

words=sorted(word_counts, key=word_counts.get,
                      reverse=True) 
print(words[:10])

# 文本的长度
text_length=len(text)
# 独特词元的数量
num_unique_words=len(words)
print(f"the text contains {text_length} words")
print(f"there are {num_unique_words} unique tokens")
# 将词元映射到索引 
word_to_int={v:k for k,v in enumerate(words)}
# 将索引映射到词元
int_to_word={k:v for k,v in enumerate(words)}
print({k:v for k,v in word_to_int.items() if k in words[:10]})
print({k:v for k,v in int_to_word.items() if v in words[:10]})

print(text[0:20])
wordidx=[word_to_int[w] for w in text]  
print([word_to_int[w] for w in text[0:20]])  

import torch

# 每个输入包含 100 个索引
seq_len=100  
xys=[]
# 从文本中的第一个词元开始，一次向右滑动一个词元
for n in range(0, len(wordidx)-seq_len-1):
    # 输入 x
    x = wordidx[n:n+seq_len]
    # 将输入 x 向右移动一个词元，作为输出 y
    y = wordidx[n+1:n+seq_len+1]
    xys.append((torch.tensor(x),(torch.tensor(y))))

from torch.utils.data import DataLoader

batch_size=32
loader = DataLoader(xys, batch_size=batch_size, shuffle=True)

x,y=next(iter(loader))
print(x)
print(y)
print(x.shape,y.shape)

import torch
from torch import nn
device="cuda" if torch.cuda.is_available() else "cpu"
class WordLSTM(nn.Module):
    def __init__(self, input_size=128, n_embed=128,
             n_layers=3, drop_prob=0.2):
        super().__init__()
        self.input_size=input_size
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_embed = n_embed
        vocab_size=len(word_to_int)
        # 训练数据首先通过嵌入层
        self.embedding=nn.Embedding(vocab_size,n_embed)
        # 创建 LSTM 层
        self.lstm = nn.LSTM(input_size=self.input_size,
            hidden_size=self.n_embed,
            num_layers=self.n_layers,
            dropout=self.drop_prob,batch_first=True)
        self.fc = nn.Linear(input_size, vocab_size)
    def forward(self, x, hc):
        embed=self.embedding(x)
        # 在每个时间步，LSTM 层利用前一个词元和隐藏状态来预测下一个词元和下一个隐藏状态
        x, hc = self.lstm(embed, hc)
        x = self.fc(x)
        return x, hc
    # 为输入序列中的第一个词元初始化隐藏状态 
    def init_hidden(self, n_seqs):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers,
                           n_seqs, self.n_embed).zero_(),
                weight.new(self.n_layers,
                           n_seqs, self.n_embed).zero_()) 
    
model=WordLSTM().to(device)
print(model)

lr=0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

model.train()

for epoch in range(50):
    tloss=0
    sh,sc = model.init_hidden(batch_size)
    # 遍历训练数据中的所有 (x, y) 批次
    for i, (x,y) in enumerate(loader):    
        if x.shape[0]==batch_size:
            inputs, targets = x.to(device), y.to(device)
            optimizer.zero_grad()
            # 预测输出序列
            output, (sh,sc) = model(inputs, (sh,sc))
            # 将预测结果与实际输出进行比较，并计算损失
            loss = loss_func(output.transpose(1,2),targets)
            sh,sc=sh.detach(),sc.detach()
            # 反向传播
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            tloss+=loss.item()
        if (i+1)%1000==0:
            print(f"at epoch {epoch} iteration {i+1}\
            average loss = {tloss/(i+1)}")

import pickle

torch.save(model.state_dict(),"files/wordLSTM.pth")
with open("files/word_to_int.p","wb") as fb:    
    pickle.dump(word_to_int, fb) 

import pickle
model.load_state_dict(torch.load("files/wordLSTM.pth"))
with open("files/word_to_int.p","rb") as fb:    
    word_to_int = pickle.load(fb)      
int_to_word={v:k for k,v in word_to_int.items()}

import numpy as np
def sample(model, prompt, length=200):
    model.eval()
    text = prompt.lower().split(' ')
    hc = model.init_hidden(1)
    # 确定需要生成多少个词元
    length = length - len(text)
    # 输入是当前序列；如果序列长度超过 100 个词元，则进行截断
    for i in range(0, length):
        if len(text)<=seq_len:
            x = torch.tensor([[word_to_int[w] for w in text]])
        else:
            x = torch.tensor([[word_to_int[w] for w in text[-seq_len:]]])            
        inputs = x.to(device)
        # 使用训练好的模型进行预测
        output, hc = model(inputs, hc)
        logits = output[0][-1]
        p = nn.functional.softmax(logits, dim=0).detach().cpu().numpy()
        # 根据预测的概率选择下一个词元
        idx = np.random.choice(len(logits), p=p)
        # 将预测的下一个词元添加到序列中
        text.append(int_to_word[idx])
    text=" ".join(text)
    for m in ",.:;?!$()/_&%*@'`":
        text=text.replace(f" {m}", f"{m} ")
    text=text.replace('"  ', '"')   
    text=text.replace("'  ", "'")  
    text=text.replace('" ', '"')   
    text=text.replace("' ", "'")     
    return text

import torch
print(sample(model, prompt='Anna and the prince'))  

def generate(model, prompt, top_k=None, 
             length=200, temperature=1):
    model.eval()
    text = prompt.lower().split(' ')
    hc = model.init_hidden(1)
    length = length - len(text)    
    for i in range(0, length):
        if len(text)<=seq_len:
            x = torch.tensor([[word_to_int[w] for w in text]])
        else:
            x = torch.tensor([[word_to_int[w] for w in text[-seq_len:]]])    
        inputs = x.to(device)
        output, hc = model(inputs, hc)
        logits = output[0][-1]
        # 通过 temperature 参数缩放 logits
        logits = logits/temperature
        p = nn.functional.softmax(logits, dim=0).detach().cpu()    
        if top_k is None:
            idx = np.random.choice(len(logits), p=p.numpy())
        else:
            # # 只保留 K 个最可能的候选项
            ps, tops = p.topk(top_k)
            ps=ps/ps.sum()
            # 从前 K 个候选项中选择下一个词元
            idx = np.random.choice(tops, p=ps.numpy())          
        text.append(int_to_word[idx])
    text=" ".join(text)
    for m in ",.:;?!$()/_&%*@'`":
        text=text.replace(f" {m}", f"{m} ")
    text=text.replace('"  ', '"')   
    text=text.replace("'  ", "'")  
    text=text.replace('" ', '"')   
    text=text.replace("' ", "'")     
    return text

prompt="I ' m not going to see"
for _ in range(10):
    print(generate(model, prompt, top_k=None, 
         length=len(prompt.split(" "))+1, temperature=1))

prompt="I ' m not going to see"
for _ in range(10):
    print(generate(model, prompt, top_k=3, 
         length=len(prompt.split(" "))+1, temperature=0.5))

print(generate(model, prompt='Anna and the prince',
               top_k=3,
               temperature=0.5)) 

prompt="I ' m not going to see"
for _ in range(10):
    print(generate(model, prompt, top_k=None, 
         length=len(prompt.split(" "))+1, temperature=2))

