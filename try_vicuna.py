import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "lmsys/vicuna-7b-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

prompt = "Explain the concept of diffusion models in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Input tokens:", inputs["input_ids"][0].tolist()[:20])
print("Input decoded:", tokenizer.decode(inputs["input_ids"][0]))
print()

outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)

print("Output tokens:", outputs[0].tolist()[:30])
print("Input length:", len(inputs["input_ids"][0]))
print("Output length:", len(outputs[0]))
print()

# Decode full output
full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Full output:", full_text[:200])
print()

# Decode only generated part
generated_ids = outputs[0][len(inputs["input_ids"][0]):]
print("Generated tokens:", generated_ids.tolist()[:20])
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("Generated only:", generated_text[:200])
