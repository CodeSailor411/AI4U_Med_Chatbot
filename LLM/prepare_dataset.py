from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Load the MedQA dataset
dataset = load_dataset("medalpaca/medical_meadow_medqa")

# Initialize the tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Define a tokenization function
def tokenize_function(examples):
    # Concatenate 'input' and 'instruction' fields for each example in the batch
    combined_texts = [i + ' ' + instr for i, instr in zip(examples['input'], examples['instruction'])]
    return tokenizer(combined_texts, padding="max_length", truncation=True, max_length=512)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset into train, validation, and test sets
def split_dataset(dataset, split_ratio=(0.8, 0.1, 0.1)):
    train_size = int(len(dataset) * split_ratio[0])
    val_size = int(len(dataset) * split_ratio[1])
    test_size = len(dataset) - train_size - val_size

    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, len(dataset)))

    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

# Split the dataset
split_datasets = split_dataset(tokenized_dataset['train'])
split_datasets.save_to_disk("split_medqa_dataset")
