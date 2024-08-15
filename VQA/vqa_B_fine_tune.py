from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer, AutoProcessor
from PIL import Image
import torch

# Load VQA dataset
dataset = load_dataset("flaviagiammarino/vqa-rad")

# Initialize the processor and model
processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-2_6")
model = AutoModel.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True,
                                   attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2_6")

# Custom preprocessing function
def preprocess_function(examples):
    images = [Image.open(image_path).convert('RGB') for image_path in examples['image']]  # Load images
    inputs = processor(
        text=examples['question'],
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    labels = tokenizer(
        text=examples['answer'],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).input_ids
    
    # Prepare inputs
    inputs['labels'] = labels
    for k, v in inputs.items():
        inputs[k] = v.squeeze(0).to(model.device)
    
    return inputs

# Preprocess the dataset
processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,  # Adjust based on GPU memory
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['test'],
)

# Fine-tune the model
trainer.train()

# Save the model and processor
model.save_pretrained("./fine-tuned-vqa-model")
tokenizer.save_pretrained("./fine-tuned-vqa-model")
