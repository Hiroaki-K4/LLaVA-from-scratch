import pickle
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from llava_image_downloader import LLaVAImageDownloader


def get_dataloader(tokenizer, batch_size=8, split="train", validation_size=1000):
    """
    Create a DataLoader for LLaVA training or validation.

    For validation: Creates a fixed validation set (cached on disk) from the first N samples.
    For training: Uses streaming dataset to avoid loading all data into memory.

    Args:
        tokenizer: Tokenizer for text processing
        batch_size: Batch size
        split: "train" or "validation"
        validation_size: Number of samples to use for validation set
    """
    validation_cache_path = Path("validation_samples.pkl")

    if split == "validation":
        # Load or create fixed validation set
        if validation_cache_path.exists():
            print(f"Loading cached validation set from {validation_cache_path}")
            with open(validation_cache_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            # First time: create validation set from streaming dataset
            print(f"Creating validation set ({validation_size} samples)...")
            stream = load_dataset(
                "liuhaotian/LLaVA-Instruct-150K", split="train", streaming=True
            )
            dataset = []
            for i, sample in enumerate(stream):
                if i >= validation_size:
                    break
                dataset.append(sample)

            # Cache for future use
            with open(validation_cache_path, "wb") as f:
                pickle.dump(dataset, f)
            print(f"Saved {len(dataset)} validation samples to cache")

        # No augmentation for validation
        transform = transforms.Compose(
            [
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    elif split == "train":
        # Use streaming dataset for training
        dataset = load_dataset(
            "liuhaotian/LLaVA-Instruct-150K", split="train", streaming=True
        )

        # Data augmentation for training
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(336, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'validation'.")

    downloader = LLaVAImageDownloader()

    system_msg = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    def collate_fn(batch):
        images = []
        all_texts = []
        all_labels_info = []  # (system_text, segments)

        for item in batch:
            image_filename = item.get("image", "")
            if not image_filename:
                continue

            # Download and preprocess image
            pil_image = downloader.download_image_by_filename(image_filename)
            if pil_image is None:
                continue

            image_tensor = transform(pil_image)
            images.append(image_tensor)

            # Build text and track segments
            system_text = system_msg + "\n"
            segments = []  # (text, is_assistant_response)
            conversations = item.get("conversations", [])

            for conv in conversations:
                if conv["from"] == "human":
                    text_value = conv["value"]
                    # Remove <image> tokens with newlines
                    text_value = text_value.replace("\n<image>", "").replace(
                        "<image>\n", ""
                    )
                    # Remove any remaining <image> tokens
                    text_value = text_value.replace("<image>", "")
                    # Clean up whitespace
                    text_value = text_value.strip()
                    text_part = f"### Human: {text_value}\n"
                    segments.append((text_part, False))
                elif conv["from"] == "gpt":
                    text_part = f"### Assistant: {conv['value']}\n"
                    segments.append((text_part, True))

            # Create full text
            full_text = system_text + "".join([s[0] for s in segments])
            all_texts.append(full_text)
            all_labels_info.append((system_text, segments))

        if len(images) == 0:
            return None

        # Tokenize all texts at once with padding
        tokenized = tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"]  # (batch_size, seq_len)
        attention_mask = tokenized["attention_mask"]

        # Create labels (mask everything initially)
        labels = input_ids.clone()
        labels[:] = -100

        # Unmask only assistant responses
        for i, (system_text, segments) in enumerate(all_labels_info):
            # Calculate token offset (BOS token if exists)
            offset = 1 if tokenizer.bos_token_id is not None else 0

            # Get system message length in tokens
            system_tokens = tokenizer(system_text, add_special_tokens=False)[
                "input_ids"
            ]
            current_pos = offset + len(system_tokens)

            # Process each segment
            for text_part, is_assistant in segments:
                part_tokens = tokenizer(text_part, add_special_tokens=False)[
                    "input_ids"
                ]
                part_len = len(part_tokens)

                if is_assistant:
                    # Keep labels for assistant responses
                    labels[i, current_pos : current_pos + part_len] = input_ids[
                        i, current_pos : current_pos + part_len
                    ]

                current_pos += part_len

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": torch.stack(images),
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=True,  # Exclude incomplete batches
    )
    return loader


if __name__ == "__main__":
    llm_id = "lmsys/vicuna-7b-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Test train loader
    print("=" * 50)
    print("Testing Train Loader (streaming)")
    print("=" * 50)
    train_loader = get_dataloader(tokenizer, batch_size=2, split="train")

    for i, batch in enumerate(train_loader):
        if batch is None:
            continue

        print(f"\n=== Batch {i} ===")
        print(f"pixel_values shape: {batch['pixel_values'].shape}")
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"labels shape: {batch['labels'].shape}")
        print(f"attention_mask shape: {batch['attention_mask'].shape}")

        if i >= 1:
            break

    # Test validation loader
    print("\n" + "=" * 50)
    print("Testing Validation Loader (cached)")
    print("=" * 50)
    val_loader = get_dataloader(
        tokenizer, batch_size=2, split="validation", validation_size=100
    )

    for i, batch in enumerate(val_loader):
        if batch is None:
            continue

        print(f"\n=== Batch {i} ===")
        print(f"pixel_values shape: {batch['pixel_values'].shape}")
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"labels shape: {batch['labels'].shape}")
        print(f"attention_mask shape: {batch['attention_mask'].shape}")

        if i >= 1:
            break

    print("\n" + "=" * 50)
    print("DataLoader test completed successfully!")
    print("=" * 50)
