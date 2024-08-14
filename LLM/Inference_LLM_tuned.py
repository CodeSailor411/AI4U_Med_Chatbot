from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

class Chatbot:
    def __init__(self, model_id, summarizer_id):
        # Initialize the main model and tokenizer with left padding
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Define pad_token_id if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize summarization model and tokenizer
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_id)
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_id)

        self.messages = [
            {"role": "system", "content": "You are a medical chatbot who helps doctors in their diagnosis."}
        ]

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def summarize_history(self, max_length=512):
        conversation_text = " ".join([msg["content"] for msg in self.messages])
        inputs = self.summarizer_tokenizer(conversation_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.summarizer_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        summary = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def generate_response(self):
        # Generate a summarized context
        summarized_context = self.summarize_history()

        # Include summarized context with the latest user message
        self.messages = [
            {"role": "system", "content": "You are a medical chatbot specialized mostly in neurology who helps doctors in their diagnosis and can suggest treatments."},
            {"role": "user", "content": summarized_context},
            {"role": "user", "content": self.messages[-1]["content"]}
        ]

        input_ids = self.tokenizer(
            [msg["content"] for msg in self.messages],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)

        eos_token_id = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            input_ids['input_ids'],
            max_new_tokens=256,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            attention_mask=input_ids['attention_mask']  # Ensure attention mask is passed
        )
        response = outputs[0][input_ids['input_ids'].shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

# Example usage
model_id = "fine-tuned-medqa-llama"  # Use your fine-tuned model ID here
summarizer_id = "facebook/bart-large-cnn"  # Example summarization model
chatbot = Chatbot(model_id, summarizer_id)

mssge = input("Please type your prompt or type 'example' to get the default example prompt: ")
# Adding user input
if mssge.lower() == "example":
    print("Your prompt is the default one: John Doe, a 45-year-old male, has a persistent cough for two weeks, shortness of breath when lying down, and swelling in his legs and ankles.\nHis medical history includes hypertension, Type 2 diabetes, mild asthma, and a past smoking habit. He is allergic to penicillin.\nBased on this information, what is the most likely diagnosis for John Doe's current condition, and what further steps or tests would you recommend?")
    chatbot.add_message("user", "John Doe, a 45-year-old male, has a persistent cough for two weeks, shortness of breath when lying down, and swelling in his legs and ankles. His medical history includes hypertension, Type 2 diabetes, mild asthma, and a past smoking habit. He is allergic to penicillin. Based on this information, what is the most likely diagnosis for John Doe's current condition, and what further steps or tests would you recommend?")
else:
    chatbot.add_message("user", mssge)

# Generate and print response
response = chatbot.generate_response()
print(response)
