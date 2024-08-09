import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from PIL import Image
from transformers import AutoFeatureExtractor

# Load the VQA dataset
dataset = load_dataset("flaviagiammarino/vqa-rad")

# Initialize feature extractor and tokenizer
feature_extractor = AutoFeatureExtractor.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5")
tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5")

# Define a function to preprocess the data
def preprocess_function(examples):
    images = [Image.open(image_path).convert("RGB") for image_path in examples['image']]
    questions = examples['question']
    answers = examples['answer']

    # Process images
    inputs = feature_extractor(images=images, return_tensors="pt")

    # Tokenize questions
    question_encodings = tokenizer(questions, truncation=True, padding=True)

    # Tokenize answers
    answer_encodings = tokenizer(answers, truncation=True, padding=True)

    return {
        'pixel_values': inputs['pixel_values'],
        'input_ids': question_encodings['input_ids'],
        'attention_mask': question_encodings['attention_mask'],
        'labels': answer_encodings['input_ids']
    }

# Preprocess the dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Initialize model
model = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM-Llama3-V-2_5")

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    save_total_limit=2
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("./fine-tuned-vqa-model")
