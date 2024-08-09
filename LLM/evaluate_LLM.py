import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_from_disk

# Load the fine-tuned model and tokenizer
model_id = "fine-tuned-medqa-llama"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the test dataset
dataset = load_from_disk("split_medqa_dataset")
test_dataset = dataset['test']

# Define a collate function
def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    import torch
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    labels = [torch.tensor(item['input_ids']) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

# Evaluate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

total_test_loss = 0

for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss}")

# Save the evaluation results
with open("evaluation_results.txt", "w") as f:
    f.write(f"Test Loss: {avg_test_loss}\n")
