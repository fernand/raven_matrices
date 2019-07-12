import torch
torch.cuda.set_device(0)
torch.cuda.manual_seed(12345)

from torch.utils.data import DataLoader, RandomSampler
from dataset import dataset

PATH = '/home/fernand/raven/neutral_pth/'
train = dataset(PATH, 'train')
valid = dataset(PATH, 'val')
test = dataset(PATH, 'test')

trainloader = DataLoader(train, batch_size=32, shuffle=True, num_workers=6)
#trainloader = DataLoader(train, batch_size=32, sampler=RandomSampler(train, replacement=True, num_samples=3200), shuffle=False, num_workers=6)
validloader = DataLoader(valid, batch_size=32, shuffle=False, num_workers=6)
testloader = DataLoader(test, batch_size=32, shuffle=False, num_workers=6)

from functools import partial
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from torch.optim import Adam

from loss import loss_fn, Accuracy
from wren import WReN

db = DataBunch(train_dl=trainloader, valid_dl=validloader, test_dl=testloader)
wren = WReN()
opt = partial(Adam, betas=(0.9, 0.999), eps=1e-8)
learn = Learner(data=db, model=wren, opt_func=opt, loss_func=loss_fn, metrics=[Accuracy()])
#from fastai.train import to_fp16
#learn = to_fp16(learn)
learn.fit(20, lr=1e-4, wd=0.0)
