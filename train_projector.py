import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from model import LlavaModel
from projection_dataloader import get_projection_data_loader

INSTRUCTION_PROMPTS = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]


def prepare_inputs_and_labels(captions, tokenizer, device, max_len=None):
    """
    Prepare input_ids and labels from captions
    """
    # Randomly select prompts for each caption
    prompts = [random.choice(INSTRUCTION_PROMPTS) for _ in captions]

    # Prepare inputs and labels
    input_ids_list = []
    labels_list = []

    for prompt, caption in zip(prompts, captions):
        # Tokenize prompt
        prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_tokens.input_ids[0]

        # Tokenize caption
        caption_tokens = tokenizer(
            caption, return_tensors="pt", add_special_tokens=False
        )
        caption_ids = caption_tokens.input_ids[0]

        # Combine prompt + caption
        input_ids = torch.cat([prompt_ids, caption_ids])

        # Labels: ignore prompt part (-100), predict caption part
        labels = torch.cat([torch.full_like(prompt_ids, -100), caption_ids])

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    # Pad sequences to same length
    max_seq_len = max(len(ids) for ids in input_ids_list)
    if max_len is not None:
        max_seq_len = min(max_seq_len, max_len)

    input_ids_padded = []
    labels_padded = []

    for input_ids, labels in zip(input_ids_list, labels_list):
        # Truncate if too long
        if max_len is not None and len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]

        pad_len = max_seq_len - len(input_ids)
        input_ids_padded.append(
            torch.cat([input_ids, torch.full((pad_len,), tokenizer.pad_token_id)])
        )
        labels_padded.append(torch.cat([labels, torch.full((pad_len,), -100)]))

    input_ids = torch.stack(input_ids_padded).to(device)
    labels = torch.stack(labels_padded).to(device)

    return input_ids, labels


@torch.no_grad()
def evaluate(model, val_loader, tokenizer, device):
    model.eval()
    total_loss = 0
    count = 0

    for images, captions in val_loader:
        input_ids, labels = prepare_inputs_and_labels(captions, tokenizer, device)
        images = images.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids, images, labels=labels)
            loss = outputs.loss

        if loss is not None:
            total_loss += loss.item()
            count += 1
        else:
            print(f"Warning: loss is None in evaluate")
            print(f"input_ids shape: {input_ids.shape}")
            print(f"labels shape: {labels.shape}")
            print(f"images shape: {images.shape}")
            print(f"outputs keys: {dir(outputs)}")

    if count == 0:
        print("Warning: No valid batches processed in evaluate")
        return float("inf")

    model.train()
    return total_loss / count


def train_projector(
    llm_id, vision_id, batch_size, num_epochs, lr_rate, patience, eval_interval, device
):
    print("Loading models...")
    model = LlavaModel(llm_id, vision_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=lr_rate)

    train_loader = get_projection_data_loader(
        batch_size=batch_size, num_workers=4, epoch_steps=100000, split="train"
    )

    val_loader = get_projection_data_loader(
        batch_size=batch_size, num_workers=4, epoch_steps=200, split="validation"
    )

    # Parameters for early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    print("Starting Training...")

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, batch_data in enumerate(pbar):
            try:
                images, captions = batch_data
                input_ids, labels = prepare_inputs_and_labels(
                    captions, tokenizer, device, max_len=128
                )
                images = images.to(device)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids, images, labels=labels)
                    loss = outputs.loss

                if loss is None:
                    print(f"Warning: loss is None at step {i}")
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"train_loss": loss.item()})

                if (i + 1) % eval_interval == 0:
                    val_loss = evaluate(model, val_loader, tokenizer, device)
                    print(f"\nStep {i+1} | Val Loss: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(model.projector.state_dict(), "best_projector.pth")
                        print("New best model saved!")
                    else:
                        patience_counter += 1
                        print(
                            f"No improvement. Patience: {patience_counter}/{patience}"
                        )

                    if patience_counter >= patience:
                        print("Early stopping triggered!")
                        return

            except (OSError, IOError, RuntimeError) as e:
                print(f"\nError at step {i}: {type(e).__name__}")
                print("Skipping this batch and continuing...")
                continue


if __name__ == "__main__":
    llm_id = "lmsys/vicuna-7b-v1.5"
    vision_id = "openai/clip-vit-large-patch14-336"
    batch_size = 4
    num_epochs = 1
    lr_rate = 1e-5
    patience = 3
    eval_interval = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    train_projector(
        llm_id,
        vision_id,
        batch_size,
        num_epochs,
        lr_rate,
        patience,
        eval_interval,
        device,
    )
