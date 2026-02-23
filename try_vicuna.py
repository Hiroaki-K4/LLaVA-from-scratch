import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "lmsys/vicuna-7b-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

prompt = "Explain the concept of diffusion models in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
