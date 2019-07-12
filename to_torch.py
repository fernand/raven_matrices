import glob
import os
import numpy as np
import torch
from scipy import misc
from multiprocessing import Pool
import io
import gzip

DIR = '/home/fernand/raven/neutral/'
OUTDIR = '/home/fernand/raven/neutral_pth/'

def process(f):
    data = np.load(f)
    image = data["image"].reshape(16, 160, 160)
    resize_image = []
    for idx in range(0, 16):
        resize_image.append(misc.imresize(image[idx,:,:], (80, 80)))
    resize_image = np.stack(resize_image)
    target = data["target"]
    meta_target = data["meta_target"]
    if meta_target.dtype == np.int8:
        meta_target = meta_target.astype(np.uint8)
    del data
    resize_image = torch.tensor(resize_image, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.long)
    meta_target = torch.tensor(meta_target, dtype=torch.float32)
    fname = f.split('.')[0].split('/')[-1]
    fname = OUTDIR + fname
    with gzip.open(fname+'_img.pth', 'wb') as w:
        buffer = io.BytesIO()
        torch.save(resize_image, buffer)
        w.write(buffer.getbuffer())
    torch.save(target, fname+'_target.pth')
    torch.save(meta_target, fname+'_mtarget.pth')

p = Pool(6)
p.map(process, glob.glob(os.path.join(DIR,'*.npz')))
