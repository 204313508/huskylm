from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto
model_name_or_path = "./model"
model = AutoModelForCausalLM.from_pretrained(
  model_name_or_path,
  torch_dtype="auto",
  device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

prefix = "What is GraphRAG?"
model_inputs = tokenizer([prefix], return_tensors="pt").to(device)

generated_ids = model.generate(
  model_inputs.input_ids,
  max_new_tokens=400,
  repetition_penalty=1.15
)
generated_ids = [
  output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split("..")[0]+"."
print("user:",prefix)
print("response:",response)