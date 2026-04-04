import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# Set pad_token to avoid warnings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, device_map="auto"
)

prompt = "Explain the concept of diffusion models in simple terms."

# Tokenize with attention mask
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
token_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs["attention_mask"].to(model.device)

outputs = model.generate(
    token_ids,
    attention_mask=attention_mask,
    max_new_tokens=200,
    temperature=0.7,
    pad_token_id=tokenizer.pad_token_id,
)

output = tokenizer.decode(outputs[0][token_ids.size(1) :], skip_special_tokens=True)
print(output)
