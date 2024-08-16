import json
from datasets import Dataset
# Import components from transformers that are needed for fine-tuning LLMs
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load the dataset from JSON file
with open('email_dataset.json', 'r') as file:
    data = json.load(file)

# Convert JSON data to a Hugging Face Dataset
dataset = Dataset.from_dict({
    'email': [item['email'] for item in data],
    'keywords': [item['keywords'] for item in data]
})

# Load pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set pad_token_id to eos_token_id as GPT-2 does not have a pad token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def preprocess_function(examples):
    inputs = [f"Email: {email} Keywords:" for email in examples['email']]
    targets = examples['keywords']
    
    # Tokenize inputs and targets with padding to the longest sequence
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments with updated evaluation_strategy
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    eval_strategy="epoch",           # evaluate after each epoch
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=2,   # batch size for training
    per_device_eval_batch_size=2,    # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    save_steps=500,
    save_total_limit=2
)

# Use a data collator that handles padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,                       # Masked language modeling is not being used and hence set to False
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_dataset,     # training dataset
    eval_dataset=tokenized_dataset,      # evaluation dataset (use a separate validation set if available)
    data_collator=data_collator          # data collator that handles padding
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

