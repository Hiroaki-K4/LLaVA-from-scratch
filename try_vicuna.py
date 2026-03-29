import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "lmsys/vicuna-7b-v1.5"

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

# print("Input tokens:", inputs["input_ids"][0].tolist()[:20])

# Decode each token individually
# print("\nIndividual token decoding:")
# for i, token_id in enumerate(inputs["input_ids"][0][:17].tolist()):
#     token_text = tokenizer.decode([token_id])
#     print(f"  Token {i}: {token_id} -> '{token_text}'")

# print("\nInput decoded (full):", tokenizer.decode(inputs["input_ids"][0]))
# print()

# outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
outputs = model.generate(
    token_ids,
    attention_mask=attention_mask,
    max_new_tokens=200,
    temperature=0.7,
    pad_token_id=tokenizer.pad_token_id,
)


# print("Output tokens:", outputs[0].tolist()[:30])
# print("Input length:", len(inputs["input_ids"][0]))
# print("Output length:", len(outputs[0]))
# print()

# Decode full output
# full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("Full output:", full_text[:200])
# print()

# Decode only generated part
# generated_ids = outputs[0][len(inputs["input_ids"][0]):]
# print("Generated tokens:", generated_ids.tolist()[:20])
# generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
# print("Generated only:", generated_text[:200])
output = tokenizer.decode(outputs[0][token_ids.size(1):], skip_special_tokens=True)
print(output)
