import os
import glob
import gzip
import io

import torch
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, root_dir, dataset_type):
        super(dataset, self).__init__()
        self.init_kwargs = {}
        self.root_dir = root_dir
        self.file_names = [f for f in glob.glob(os.path.join(root_dir, "*img.pth")) \
                            if dataset_type in f]
        print(dataset_type, len(self.file_names))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = self.file_names[idx]
        with gzip.open(img_path, 'rb') as f:
            data = torch.load(io.BytesIO(f.read()))
        base = '_'.join(img_path.split('_')[:-1])
        target = torch.load(base + '_target.pth')
        meta_target = torch.load(base + '_mtarget.pth')
        return data, (target, meta_target)

