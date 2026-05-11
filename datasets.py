import torch
from torch.utils.data import Dataset
import numpy as np
import os

class PretokenizedDataset(Dataset):
    def __init__(self, data_dir: str, split: str, block_size: int):
        """
        A generic dataset for reading pre-tokenized numpy memmap files.
        
        Args:
            data_dir: Path to the directory containing the .bin files.
            split: 'train' or 'val' (used to construct the filename).
            block_size: The context length of the model.
        """
        super().__init__()
        self.block_size = block_size
        self.filepath = os.path.join(data_dir, f"{split}.bin")
        
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Could not find {self.filepath}")
        
        # Calculate length using file size on disk (np.uint16 = 2 bytes)
        file_size_bytes = os.path.getsize(self.filepath)
        total_tokens = file_size_bytes // 2
        self.length = total_tokens - self.block_size
        
        # Lazy Initialization for DataLoader multiprocessing
        self.data = None

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.data is None:
            self.data = np.memmap(self.filepath, dtype=np.uint16, mode='r')
            
        chunk = self.data[idx : idx + self.block_size + 1]
    
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y