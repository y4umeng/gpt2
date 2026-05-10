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
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Load the data using memmap for efficiency
        self.data = np.memmap(filepath, dtype=np.uint16, mode='r')

    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
    
        # Convert to torch tensor
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y

def train():
    config = GPTConfig(vocab_size=50304)
    batch_size = 16
    epochs = 1
    device = 'cuda'
    num_workers = 4
    
    train_ds = ShakespeareDataset('train', config.block_size)
    val_ds = ShakespeareDataset('val', config.block_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    print(f"Loaded train set of {len(train_dl)} batches, and val set of {len(val_dl)} batches")

    torch.set_float32_matmul_precision('high')

    model = GPT(config)
    model.to(device)
    model = torch.compile(model)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    lr_scheduler = 0

    print("Training!")
    for e in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_dl):
            t1 = time.time()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model.forward(x, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            torch.cuda.synchronize()
            t2 = time.time()
            dt = (t2 - t1)*1000
            print(f"step {i} | loss: {loss.item():.6f} | grad norm: {norm:.4f} | time {dt:.2f}ms | tok/sec: {(config.block_size * batch_size) / (dt/1000)}")
            if i == 20:
                sys.exit(0)

        model.eval()
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                _, loss = model.forward(x, y)
        print(f"epoch {e} done")

train()