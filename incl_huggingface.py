import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
# model_name = "gpt2"
model_name = "./trained_model"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Define few-shot examples
examples = """
Example 1:
Email: "Hi Team, The new design includes an open floor plan with large windows to maximize natural light. Regards, Sam"
Extracted Keywords: "Open floor plan", "large windows", "maximize natural light"

Example 2:
Email: "Dear Colleagues, We have finalized the blueprints for the new project. The building will feature a green roof and solar panels for sustainability. Cheers, Jamie"
Extracted Keywords: "Green roof", "solar panels", "sustainability"

Example 3:
Email: "Hello Team, The construction of the new research facility will incorporate advanced seismic safety features, including reinforced structures and vibration-dampening systems, to ensure resilience against earthquakes. Regards, Morgan"
Extracted Keywords: "seismic safety features", "reinforced structures", "vibration-dampening systems", "earthquake resilience"

Example 4:
Email: "Dear All, The design update includes a new water feature in the main lobby to create a calming atmosphere. Additionally, we’re adding natural stone flooring and modern art installations to enhance the space's aesthetic appeal. Cheers, Riley"
Extracted Keywords: "water feature", "natural stone flooring", "modern art installations", "calming atmosphere", "aesthetic appeal"


"""

# Define the actual email content
email_content = """
Hi Team,

Please find the latest updates on the new building design. We have incorporated the feedback on the structural integrity and made adjustments to the foundation layout. The new plans include reinforced concrete supports and optimized load distribution.

Best,
Alex
"""

# Combine examples and the actual email content into the prompt
prompt = f"""
{examples}

Email: "{email_content}"
Extracted Keywords:
"""

# Tokenize input with attention mask
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Generate the completion
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,  # Add the attention mask here
    max_length=600,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,
    no_repeat_ngram_size=2,
    top_k=50,
    do_sample=True
)

# Decode the output and print it
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
if output_text:
    print("Extracted Knowledge:", output_text)
else:
    print("No meaningful output was generated. Please refine the prompt or adjust the parameters.")
