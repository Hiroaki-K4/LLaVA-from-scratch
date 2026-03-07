import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, CLIPVisionModel


class LlavaModel(nn.Module):
    def __init__(self, llm_model_name, vision_model_name, projector_path=None):
        super().__init__()
        # Vision encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_encoder.requires_grad_(False)

        # Language model
        self.language_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.language_model.requires_grad_(False)

        # Projection layers
        vision_hidden_size = self.vision_encoder.config.hidden_size
        llm_hidden_size = self.language_model.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        if projector_path is not None:
            self.projector.load_state_dict(torch.load(projector_path))

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values, output_hidden_states=True)
            image_features = outputs.hidden_states[-2][:, 1:]

        image_features = self.projector(image_features)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        combined_embeds = torch.cat([image_features, input_embeds], dim=1)

        # Prepare attention_mask
        if attention_mask is not None:
            batch_size, img_seq_len, _ = image_features.shape
            # Image features are always attended to (all 1s)
            image_attention = torch.ones(
                (batch_size, img_seq_len),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([image_attention, attention_mask], dim=1)

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
            inputs_embeds=combined_embeds.to(torch.float16),
            attention_mask=attention_mask,
            labels=labels,
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
    dummy_attention_mask = torch.ones(1, 16, dtype=torch.long)

    dummy_ids = dummy_ids.to(device)
    dummy_pixels = dummy_pixels.to(device)
    dummy_labels = dummy_labels.to(device)
    dummy_attention_mask = dummy_attention_mask.to(device)

    outputs = model(
        dummy_ids,
        dummy_pixels,
        attention_mask=dummy_attention_mask,
        labels=dummy_labels,
    )
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")
