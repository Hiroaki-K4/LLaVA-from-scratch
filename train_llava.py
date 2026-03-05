import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from model import LlavaModel


def train_llava(
    llm_id,
    llm_requires_grad,
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
    model = LlavaModel(llm_id, vision_id, llm_requires_grad, projector_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    tokenizer.pad_token = tokenizer.eos_token
    print(model)


if __name__ == "__main__":
    llm_id = "lmsys/vicuna-7b-v1.5"
    llm_requires_grad = True
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
        llm_requires_grad,
        vision_id,
        projector_path,
        batch_size,
        num_epochs,
        lr_rate,
        patience,
        eval_interval,
        device,
    )
