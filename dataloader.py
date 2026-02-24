import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms


def get_data_loader(batch_size=32, num_workers=4, split="train"):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_url = "https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/"
    if split == "train":
        all_urls = [
            f"pipe:curl -s -L {base_url}cc3m-{split}-{i:04d}.tar" for i in range(576)
        ]
        dataset = wds.WebDataset()
    elif split == "validation":
        all_urls = [
            f"pipe:curl -s -L {base_url}cc3m-{split}-{i:04d}.tar" for i in range(16)
        ]
        dataset = wds.WebDataset()
