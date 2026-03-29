import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Llamaモデルを使用（例: Llama-2またはLlama-3）
# model_name = "meta-llama/Llama-2-7b-chat-hf"  # Llama 2 Chat
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Llama 3 8B Instruct
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Llama 3.2 1B Instruct (lightweight)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

prompt = "Explain the concept of diffusion models in simple terms."

# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

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
    token_ids.to(model.device), max_new_tokens=200, temperature=0.7
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
output = tokenizer.decode(outputs.tolist()[0][token_ids.size(1) :])
print(output)
