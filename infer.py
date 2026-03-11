import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from model import LlavaModel


def generate_response(
    model, tokenizer, image_processor, image_path, prompt_text, device
):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(image=image, return_tensors="pt")["pixel_values"].to(
        device
    )

    full_prompt = f"USER: <image>\n{prompt_text} ASSISTANT:"
    input_ids = tokenizer(full_prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        vision_outputs = model.vision_encoder(pixel_values, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2][:, 1:]

        image_features = model.projector(image_features)

        input_embeds = model.language_model.get_input_embeddings()(input_ids)
        combined_embeds = torch.cat([image_features, input_embeds], dim=1)

        batch_size, img_seq_len, _ = image_features.shape
        text_seq_len = input_ids.shape[1]
        attention_mask = torch.ones(
            (batch_size, img_seq_len + text_seq_len), device=device, dtype=torch.long
        )

        generate_ids = model.language_model.generate(
            input_embeds=combined_embeds.to(torch.float16),
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

    if "ASSISTANT:" in generated_text:
        response = generated_text.split("ASSISTANT:")[-1].strip()
    else:
        response = generated_text

    return response


def infer(llm_model_name, vision_model_name, projector_path, llava_model_path):
    model = LlavaModel(
        llm_model_name=llm_model_name,
        vision_model_name=vision_model_name,
        projector_path=projector_path,
    ).to(device)
    model.language_model = PeftModel.from_pretrained(
        model.language_model, llava_model_path
    )

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)

    image_path = "example.jpg"
    question = "What is in this image"

    print(f"\nQuestion: {question}")
    response = generate_response(
        model, tokenizer, image_processor, image_path, question, device
    )
    print(f"Answer: {response}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_model_name = "lmsys/vicuna-7b-v1.5"
    vision_model_name = "openai/clip-vit-large-patch14-336"
    projector_path = "best_projector.pth"
    llava_model_path = "best_llava.pth"
