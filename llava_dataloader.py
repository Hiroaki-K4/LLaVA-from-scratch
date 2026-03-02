import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from llava_image_downloader import LLaVAImageDownloader


def get_dataloader(tokenizer, batch_size=8, split="train"):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset = load_dataset(
        "liuhaotian/LLaVA-Instruct-150K", split=split, streaming=True
    )

    downloader = LLaVAImageDownloader()

    system_msg = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    def collate_fn(batch):
        for item in batch:
            image_filename = item.get("image", "")
            if image_filename:
                image = downloader.download_image_by_filename(image_filename)
                print(image)
                input()

                conversations = item.get("conversations", [])
                text = system_msg + "\n"
                for j, conv in enumerate(conversations):
                    if conv["from"] == "human":
                        text += f"### Human: {conv["value"]}\n"
                    elif conv["from"] == "gpt":
                        text += f"### Assistant: {conv["value"]}\n"

                text += "###"
                print(text)
                input()
                return text
            else:
                None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=True,  # Exclude incomplete batches
    )
    return loader


if __name__ == "__main__":
    test_loader = get_dataloader(None)
    for i, batch in enumerate(test_loader):
        print(i, batch)
