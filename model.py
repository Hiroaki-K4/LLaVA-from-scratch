import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, CLIPVisionModel


class LlavaModel(nn.Module):
    def __init__(self, llm_model_name, vision_model_name):
        super().__init__()
        # Vision encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_encoder.requires_grad_(False)

        # Language model
        # self.language_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.language_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            quantization_config=quantization_config,  # 追加
            device_map="auto",  # 自動的にGPUへ割り振り
        )
        self.language_model.requires_grad_(False)

        # Projection layers
        vision_hidden_size = self.vision_encoder.config.hidden_size
        llm_hidden_size = self.language_model.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, input_ids, pixel_values, labels=None):
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values, output_hidden_states=True)
            image_features = outputs.hidden_states[-2][:, 1:]

        image_features = self.projector(image_features)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        combined_embeds = torch.cat([image_features, input_embeds], dim=1)

        if labels is not None:
            batch_size, img_seq_len, _ = image_features.shape
            ignore_labels = torch.full(
                (batch_size, img_seq_len),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([ignore_labels, labels], dim=1)

        return self.language_model(
            inputs_embeds=combined_embeds.to(torch.float16), labels=labels
        )


if __name__ == "__main__":
    LLM_ID = "lmsys/vicuna-7b-v1.5"
    VISION_ID = "openai/clip-vit-large-patch14-336"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = LlavaModel(LLM_ID, VISION_ID).to(device)
    dummy_pixels = torch.randn(1, 3, 336, 336)
    dummy_ids = torch.randint(0, 32000, (1, 16))
    dummy_labels = torch.randint(0, 32000, (1, 16))

    dummy_ids = dummy_ids.to(device)
    dummy_pixels = dummy_pixels.to(device)
    dummy_labels = dummy_labels.to(device)

    outputs = model(dummy_ids, dummy_pixels, labels=dummy_labels)
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")
