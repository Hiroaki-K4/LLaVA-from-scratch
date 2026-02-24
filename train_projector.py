import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloader import get_data_loader
from model import LlavaModel


def train_projector(llm_id, vision_id, batch_size, num_epochs, lr_rate, device):
    print("Loading models...")
    model = LlavaModel(llm_id, vision_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=lr_rate)

    train_loader = get_data_loader(
        batch_size=batch_size, num_workers=4, epoch_steps=2000, split="train"
    )

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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"train_loss": loss.item()})


if __name__ == "__main__":
    llm_id = "lmsys/vicuna-7b-v1.5"
    vision_id = "openai/clip-vit-large-patch14-336"
    batch_size = 4
    num_epochs = 1
    lr_rate = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_projector(llm_id, vision_id, batch_size, num_epochs, lr_rate, device)
