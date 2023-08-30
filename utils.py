import random 
import json
import numpy as np
import random
import os
import torch
import pickle
import time
import shutil
import functools

def load_json(path):
    with open(path,'r') as f:
        res = json.load(f)
    return res


def save_json(obj, path:str):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=4)


def load_pkl(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
        return res


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def setup_seed(seed = 3407):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    # https://zhuanlan.zhihu.com/p/73711222
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_datetime():
    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    return t


def mkdir(dir:str, rm:bool=False):
    if os.path.isdir(dir):
        if rm:
            shutil.rmtree(dir)
            os.makedirs(dir)
        else:
            pass
    else:
        os.makedirs(dir)


def record_time(func):
    @functools.wraps(func)
    def run(*args, **kwds):
        torch.cuda.synchronize()
        s = time.time()
        ans = func(*args, **kwds)
        torch.cuda.synchronize()
        e = time.time()
        print(f"Running time for {func.__name__} = {e-s} (s)")
        return ans
    return run



class MaskGenerator(torch.nn.Module):
    def __init__(self, mask_ratio:float, masked_padding_value:float) -> None:
        """
        @params:
            mask_ratio: ratio to masked
            masked_padding_value, replace original value with this
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.masked_padding_value = masked_padding_value
    
    
    def forward(self, x:torch.Tensor):
        return self.generate_mask(x)

    
    def generate_mask(self, x:torch.Tensor):
        """
        @params:
            x: shape = [batch_size, seq_len, channel]
        @returns:
            masked_x
            mask: 1 means keep, 0 means masked
            mask_idx
            unmask_idx
        """
        mask_len = int(self.mask_ratio * x.shape[1])

        idx_shuffle = torch.rand_like(x).argsort(dim=1)

        # .sort() to keep their original relative order
        mask_idx = idx_shuffle[:, 0:mask_len, :].sort(dim=1)[0]
        unmask_idx = idx_shuffle[:, mask_len:, :].sort(dim=1)[0]
        restore_idx = torch.cat([mask_idx, unmask_idx], dim=1).argsort(dim=1)

        masked_part = torch.full_like(mask_idx, self.masked_padding_value, dtype=x.dtype)
        unmasked_part = x.gather(dim=1, index=unmask_idx)
        masked_x = torch.cat([masked_part, unmasked_part], dim=1).gather(dim=1, index=restore_idx)

        mask = torch.cat([torch.zeros_like(mask_idx), torch.ones_like(unmask_idx)], dim=1)
        mask = mask.gather(dim=1, index=restore_idx)

        return masked_x, mask, mask_idx, unmask_idx