import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "./trained_model"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Handles padding correctly to avoid any issues with padding token being different from the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Define few-shot examples to guide the model in extracting keywords and help it understand the format and context for keyword extraction
examples = """

Example 1:
Email: "Hello Team, We should evaluate the trade-offs between using BookKeeper and building our own commit log system. While BookKeeper offers a mature solution, it might introduce unnecessary complexity. We need to consider factors such as performance, reliability, and maintainability. Regards, Uma"
Keywords: "BookKeeper, commit log, trade-offs, performance, reliability, maintainability, complexity"

Example 2
Email: "Hi Devs, \n\nI've done some benchmarking between BookKeeper and our proposed solution. Our approach seems to outperform BookKeeper in terms of write latency, especially under high load conditions. However, BookKeeper's community support might be beneficial for long-term maintenance. Best Regards, Sujee"
Keywords: "benchmarking, performance, latency, write performance, community support, maintenance"

Example 3
Email: "Dear Team, To ensure data durability, we should consider using a replication factor of 3 for both BookKeeper and our proposed solution. We can also explore asynchronous replication as an optimization. \n\nThanks, Alex"
Keywords: "data durability, replication factor, asynchronous replication"

"""

# Define the actual email content
email_content = """
Hello Devs,

I am trying to understand the current state / direction of namenode High Availability(HA) implementation... I see the following options:
1) rely on external HA filer
2) multiple edit directories
3) book keeper

Regards,
Sujee
"""

# Combine examples and the actual email content into the prompt to provide context and format to the model for generating the output
prompt = f"""
{examples}

Email: "{email_content}"
Extracted Keywords:
"""

# Tokenize input with attention mask to convert the prompt into model-readable format and create attention masks
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Generate the completion
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,   # Add attention mask to guide the model
    max_length=600,                         # Maximum length of the generated sequence
    num_return_sequences=1,                 # Number of sequences to generate
    pad_token_id=tokenizer.eos_token_id,    # Token ID used for padding
    no_repeat_ngram_size=2,                 # Prevent repeating n-grams
    top_k=50,                               # Use top-k sampling for diversity
    do_sample=True                          # Enable sampling for generation
)

# Decode the output from token IDs to text and print it
# Remove special tokens and strip leading/trailing whitespace
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if output_text:
    print("Extracted Knowledge:", output_text)
else:
    print("No meaningful output was generated. Please refine the prompt or adjust the parameters.")
