import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloader import get_data_loader
from model import LlavaModel


@torch.no_grad()
def evaluate(model, val_loader, tokenizer, device):
    model.eval()
    total_loss = 0
    count = 0

    for images, captions in val_loader:
        text_inputs = tokenizer(
            captions, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)

        input_ids = text_inputs.input_ids
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

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
        return float('inf')

    model.train()
    return total_loss / count


def train_projector(
    llm_id, vision_id, batch_size, num_epochs, lr_rate, patience, device
):
    print("Loading models...")
    model = LlavaModel(llm_id, vision_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=lr_rate)

    train_loader = get_data_loader(
        batch_size=batch_size, num_workers=4, epoch_steps=2000, split="train"
    )

    val_loader = get_data_loader(
        batch_size=batch_size, num_workers=4, epoch_steps=200, split="validation"
    )

    # Parameters for early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    eval_interval = 200

    model.train()
    print("Starting Training...")

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (images, captions) in enumerate(pbar):
            text_inputs = tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)

            input_ids = text_inputs.input_ids
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            images = images.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids, images, labels=labels)
                loss = outputs.loss

            if loss is None:
                print(f"Warning: loss is None at step {i}")
                print(f"input_ids shape: {input_ids.shape}")
                print(f"labels shape: {labels.shape}")
                print(f"images shape: {images.shape}")
                print(f"outputs keys: {dir(outputs)}")
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
                    print(f"No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    return


if __name__ == "__main__":
    llm_id = "lmsys/vicuna-7b-v1.5"
    vision_id = "openai/clip-vit-large-patch14-336"
    batch_size = 4
    num_epochs = 1
    lr_rate = 1e-5
    patience = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_projector(
        llm_id, vision_id, batch_size, num_epochs, lr_rate, patience, device
    )
