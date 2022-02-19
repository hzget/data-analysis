from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import logging

logging.basicConfig(level=logging.INFO)
#logging.info(f'is loading AI model ...')

# train_iter = AG_NEWS(root="./data", split="train")
tokenizer = get_tokenizer("basic_english")

# Because datasets are iterators, if we want to use the
# data multiple times we need to convert it to list
train_dataset, test_dataset = AG_NEWS(root="./data")
train_dataset = list(train_dataset)
test_dataset = list(test_dataset)
train_iter = iter(train_dataset)

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# example
# vocab(["here", "is", "an", "example"])
# output: [475, 21, 30, 5297]

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1
encode = text_pipeline

import torch
vocab_size = len(vocab)
def to_bow(text, bow_vocab_size=vocab_size):
    res = torch.zeros(bow_vocab_size, dtype=torch.float32)
    for i in encode(text):
        if i < bow_vocab_size:
            res[i] += 1
    return res

# example
# b = to_bow("how are you")
# print(b, b.shape)
# output: tensor([0., 0., 0.,  ..., 0., 0., 0.]) torch.Size([95811])

from torch.utils.data import DataLoader
import numpy as np

# this collate function gets list of batch_size tuples, and needs to
# return a pair of label-feature tensors for the whole minibatch
def bowify(b):
    return (
            torch.LongTensor([t[0]-1 for t in b]),
            torch.stack([to_bow(t[1]) for t in b])
    )

# convert our dataset for training in such a way, that
# all positional vector representations are converted to
# bag-of-words representation
train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=bowify, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=bowify, shuffle=True)

net = torch.nn.Sequential(torch.nn.Linear(vocab_size,4),torch.nn.LogSoftmax(dim=1))

def train_epoch(net,dataloader,lr=0.01,optimizer=None,loss_fn = torch.nn.NLLLoss(),epoch_size=None, report_freq=200):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    net.train()
    total_loss,acc,count,i = 0,0,0,0
    for labels,features in dataloader:
        optimizer.zero_grad()

        # forward & get loss
        out = net(features)
        loss = loss_fn(out,labels) #cross_entropy(out,labels)

        # backword to get gradient
        loss.backward()

        # update params
        optimizer.step()

        # collect data to print
        total_loss+=loss
        _,predicted = torch.max(out,1)
        acc+=(predicted==labels).sum()
        count+=len(labels)
        i+=1
        if i%report_freq==0:
            print(f"{count}: acc={acc.item()/count}")
        if epoch_size and count>epoch_size:
            break
    return total_loss.item()/count, acc.item()/count

# specify small epoch_size because of low compute power
# example:
# train_epoch(net,train_loader,epoch_size=15000)
# logs and output
# 3200: acc=0.811875
# 6400: acc=0.845
# 9600: acc=0.8578125
# 12800: acc=0.863671875
# Out[14]: (0.026340738796730285, 0.8640724946695096)

classes = ['World', 'Sports', 'Business', 'Sci/Tech']

def getTextClass(text, net=net):
    net.eval()
    x = torch.stack([to_bow(text)])
    y = net(x)
    _, index = torch.max(y,1)
    return classes[index]

def save():
    global net
    torch.save(net.state_dict(), "bow.pt")

def load():
    global net
    net.load_state_dict(torch.load("bow.pt"))
    net.eval()
    return net

