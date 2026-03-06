import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer

from llava_dataloader import get_dataloader
from model import LlavaModel


def train_llava(
    llm_id,
    vision_id,
    projector_path,
    batch_size,
    num_epochs,
    lr_rate,
    patience,
    eval_interval,
    device,
):
    print("Loading models...")
    model = LlavaModel(llm_id, vision_id, projector_path).to(device)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    model.language_model = get_peft_model(model.language_model, lora_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)

    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    tokenizer.pad_token = tokenizer.eos_token
    print(model)

    train_loader = get_dataloader(
        tokenizer=tokenizer, batch_size=batch_size, split="train"
    )
    val_loader = get_dataloader(
        tokenizer=tokenizer, batch_size=batch_size, split="validation"
    )


if __name__ == "__main__":
    llm_id = "lmsys/vicuna-7b-v1.5"
    vision_id = "openai/clip-vit-large-patch14-336"
    projector_path = "best_projector.pth"
    batch_size = 4
    num_epochs = 1
    lr_rate = 1e-5
    patience = 3
    eval_interval = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_llava(
        llm_id,
        vision_id,
        projector_path,
        batch_size,
        num_epochs,
        lr_rate,
        patience,
        eval_interval,
        device,
    )
