# huskyLM-0.5B-academic-v0.1

huskyLM-0.5B-academic-v0.1 is a pre-trained language model based on the Qwen-2 tokenizer, developed using a custom-curated dataset from arXiv papers published in 2024. These papers were meticulously selected and organized by the developer to focus on research related to large language models (LLMs). The model is designed to excel in generating coherent and contextually relevant text, making it a powerful tool for various NLP tasks.

## Features

- **Model Size**: 0.5 Billion parameters
- **Training Data**: Self-collected and organized dataset from arXiv papers published in 2024, focusing on large language models.
- **Tokenizer**: Utilizes the Qwen-2 tokenizer for efficient tokenization and preprocessing.
- **Supported Tasks**: Text generation, language modeling, and more.

## Installation

To use huskyLM-0.5B-academic-v0.1, you'll need to have `transformers` and `torch` installed. You can install these packages via pip:

```bash
pip install transformers torch
```

## Usage

Here is a simple example of how to use huskyLM-0.5B-academic-v0.1 for text generation:

```python
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
```

### Explanation:

- **user_input**: The initial text prompt to guide the model's text generation.
- **max_new_tokens**: Limits the number of tokens generated in response.
- **repetition_penalty**: Adjusts the penalty for repetitive sequences, helping to maintain diversity in the generated text.
  
## License

huskyLM-0.5B-academic-v0.1 is provided under the [insert appropriate license]. Please ensure proper attribution when using this model in your projects.
