"""Validated Hugging Face CIFAR-10 adapter."""
from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_from_disk

class HFCIFAR10(Dataset):
    def __init__(self, root: str, train: bool, transform=None):
        self.ds=load_from_disk(root)["train" if train else "test"]
        self.targets=self.ds["label"]
        self.transform=transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r=self.ds[int(i)]; image=r["img"].convert("RGB")
        return (self.transform(image) if self.transform else image), int(r["label"])
