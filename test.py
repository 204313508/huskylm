from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
  "huskyhong/huskyLM-0.5B-academic-v0.1",
  torch_dtype="auto",
  device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("huskyhong/huskyLM-0.5B-academic-v0.1")

user_input = "What is GraphRAG?"
model_inputs = tokenizer([user_input], return_tensors="pt").to(device)

generated_ids = model.generate(
  model_inputs.input_ids,
  max_new_tokens=400,
  repetition_penalty=1.15
)
generated_ids = [
  output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split("..")[0]+"."
print("user:", user_input)
print("response:", response)
