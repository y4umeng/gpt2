import torch
from model import GPT, GPTConfig
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

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
    batch_size = 512 # 524288
    micro_batch_size = 64
    assert batch_size % micro_batch_size == 0 and micro_batch_size <= batch_size
    grad_accum_steps = batch_size // micro_batch_size
    device = 'cuda'
    num_workers = 4
    log_interval = 1

    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert grad_accum_steps % ddp_world_size == 0
        grad_accum_steps //= ddp_world_size
        if master_process: print(f"Training on {ddp_world_size} devices")
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        print("Training on 1 device")

    torch.manual_seed(1337 + seed_offset)
    
    train_ds = ShakespeareDataset('train', config.block_size)
    val_ds = ShakespeareDataset('val', config.block_size)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None
    train_dl = DataLoader(
        train_ds, 
        batch_size=micro_batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None), # Must be False if using a sampler
        pin_memory=True, 
        num_workers=num_workers, 
        drop_last=True
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=micro_batch_size, 
        sampler=val_sampler, 
        shuffle=(val_sampler is None), # Must be False if using a sampler
        pin_memory=True, 
        num_workers=num_workers, 
        drop_last=True
    )

    if master_process: print(f"Loaded train set of {len(train_dl) // grad_accum_steps} batches, and val set of {len(val_dl) // grad_accum_steps} batches")

    torch.set_float32_matmul_precision('high')

    model = GPT(config)
    model.to(device)
    model = torch.compile(model)
    if master_process: print(f"Total params: {model.num_params()}")
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    optim = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train_dl) // grad_accum_steps)

    print("Training!")

    model.train()
    optim.zero_grad(set_to_none=True)
    max_micro_batches = (len(train_dl) // grad_accum_steps) * grad_accum_steps
    tokens_per_iter = ddp_world_size * batch_size * config.block_size
    if master_process: print(f"tokens per iteration will be: {tokens_per_iter:,}")
    t1 = time.time()
    for i, (x, y) in enumerate(train_dl):
        if i >= max_micro_batches:
            break

        last_micro_step_of_batch = (i + 1) % grad_accum_steps == 0
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            model.require_backward_grad_sync = last_micro_step_of_batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
        if last_micro_step_of_batch:
            batch_num = (i + 1)//grad_accum_steps - 1
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            if batch_num % log_interval == 0 and master_process:
                t2 = time.time()
                t1 = t2
                dt = (t2 - t1)*1000
                dt /= log_interval
                # loss is an estimate over total batch given single micro batch
                print(f"step {batch_num} | loss: {loss.item() * grad_accum_steps:.6f} | grad norm: {norm:.4f} | time {dt:.2f}ms | tok/sec: {(tokens_per_iter) / (dt/1000):.2f}")
            if batch_num == 20:
                break

    model.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            val_loss += loss.item()
            val_steps += 1
    print(f"Validation loss: {val_loss / val_steps:.4f}")
    print(f"One pass of training set done")

    if ddp:
        destroy_process_group()

train()
print("Done training!")