from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# Load VQA dataset
dataset = load_dataset("flaviagiammarino/vqa-rad")

# Initialize tokenizer and model
model_id = "deepseek-ai/deepseek-vl-7b-chat"  # Use your base model ID
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        text=examples['text'],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test']
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("./fine-tuned-vqa-model")
tokenizer.save_pretrained("./fine-tuned-vqa-model")
