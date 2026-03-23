import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
llm_model_name = "lmsys/vicuna-7b-v1.5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name, torch_dtype=torch.float16
).to(device)

print("Loading PEFT adapter...")
model = PeftModel.from_pretrained(base_model, "best_llava")

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Test text completion
system_msg = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
user_question = "What is the capital of France?"
full_prompt = f"{system_msg}\n### Human: {user_question}\n### Assistant:"

print(f"\nPrompt: {full_prompt}\n")

inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Full response:\n{response}\n")

# Extract just the assistant's response
if "### Assistant:" in response:
    assistant_response = response.split("### Assistant:")[-1].strip()
    print(f"Assistant only:\n{assistant_response}")
