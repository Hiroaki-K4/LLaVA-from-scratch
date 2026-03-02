import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms


def get_projection_dataloader(
    batch_size=32, num_workers=4, epoch_steps=1000, split="train"
):
    preprocess = transforms.Compose(
        [
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_url = "https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/"
    
    # Add retry logic to curl command for better reliability
    curl_cmd = "curl --retry 5 --retry-delay 2 --retry-max-time 60 -s -L"
    
    if split == "train":
        all_urls = [
            f"pipe:{curl_cmd} {base_url}cc3m-{split}-{i:04d}.tar" for i in range(576)
        ]
        dataset = (
            wds.WebDataset(all_urls, resampled=True, handler=wds.warn_and_continue)
            .shuffle(2000)
            .decode("pil", handler=wds.warn_and_continue)
            .to_tuple("jpg", "txt", handler=wds.warn_and_continue)
            .map_tuple(preprocess, lambda x: x)
            .batched(batch_size)
            .with_epoch(epoch_steps)
        )
    elif split == "validation":
        all_urls = [
            f"pipe:{curl_cmd} {base_url}cc3m-{split}-{i:04d}.tar" for i in range(16)
        ]
        dataset = (
            wds.WebDataset(all_urls, resampled=False, shardshuffle=False, handler=wds.warn_and_continue)
            .shuffle(2000)
            .decode("pil", handler=wds.warn_and_continue)
            .to_tuple("jpg", "txt", handler=wds.warn_and_continue)
            .map_tuple(preprocess, lambda x: x)
            .batched(batch_size)
            .with_epoch(epoch_steps)
        )

    loader = DataLoader(
        dataset, batch_size=None, num_workers=num_workers, pin_memory=True
    )

    return loader


if __name__ == "__main__":
    loader = get_projection_dataloader(batch_size=8, num_workers=2, split="train")
    for i, (images, captions) in enumerate(loader):
        print(f"--- Batch {i+1} ---")
        print(
            f"Images shape: {images.shape}"
        )  # [batch_size, 3, image_size, image_size]
        print(f"First caption: {captions[0]}")

        if i >= 2:
            break
