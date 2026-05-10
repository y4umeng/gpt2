import torch
from model import GPT, GPTConfig
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import sys

class ShakespeareDataset(Dataset):
    def __init__(self, split: str, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.split = split
        filename = 'train.bin' if split == 'train' else 'val.bin'
        self.filepath = os.path.join(os.path.dirname(__file__), "data/shakespeare/", filename)
        
        # 1. Calculate length using file size on disk, NOT by loading the data.
        # np.uint16 takes up 2 bytes per token.
        file_size_bytes = os.path.getsize(self.filepath)
        total_tokens = file_size_bytes // 2
        
        self.length = total_tokens - self.block_size
        
        # 2. Set data to None initially (Lazy Initialization)
        self.data = None

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.data is None:
            self.data = np.memmap(self.filepath, dtype=np.uint16, mode='r')
            
        chunk = self.data[idx : idx + self.block_size + 1]
    
        # Convert to torch tensor
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y

def train():
    config = GPTConfig(vocab_size=50304)
    batch_size = 512
    micro_batch_size = 64
    assert batch_size % micro_batch_size == 0 and micro_batch_size <= batch_size
    grad_accum_steps = batch_size // micro_batch_size
    device = 'cuda'
    num_workers = 4
    
    train_ds = ShakespeareDataset('train', config.block_size)
    val_ds = ShakespeareDataset('val', config.block_size)
    train_dl = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    print(f"Loaded train set of {len(train_dl) // grad_accum_steps} batches, and val set of {len(val_dl) // grad_accum_steps} batches")

    torch.set_float32_matmul_precision('high')

    model = GPT(config)
    model.to(device)
    model = torch.compile(model)
    optim = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train_dl) // grad_accum_steps)

    print("Training!")

    model.train()
    optim.zero_grad(set_to_none=True)
    t1 = time.time()
    for i, (x, y) in enumerate(train_dl):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model.forward(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
        if (i + 1) % grad_accum_steps == 0:
            batch_num = (i + 1)//grad_accum_steps - 1
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t2 = time.time()
            dt = (t2 - t1)*1000
            print(f"step {batch_num} | loss: {loss.item():.6f} | grad norm: {norm:.4f} | time {dt:.2f}ms | tok/sec: {(config.block_size * batch_size) / (dt/1000):.2f}")
            t1 = time.time()
            if batch_num == 20:
                sys.exit(0)

    model.eval()
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _, loss = model.forward(x, y)
    print(f"One pass of training set done")

train()