from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

class Chatbot:
    def __init__(self, model_id, summarizer_id):
        # Initialize main model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Initialize summarization model and tokenizer
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_id)
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_id)

        self.messages = [
            {"role": "system", "content": "You are a medical chatbot specialized mostly in neurology who helps doctors in their diagnosis and can suggest treatments ."}
        ]

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def summarize_history(self, max_length=500):
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
            {"role": "system", "content": "You are a medical chatbot specialized mostly in neurology who helps doctors in their diagnosis and can suggest treatments ."},
            {"role": "user", "content": summarized_context},
            {"role": "user", "content": self.messages[-1]["content"]}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Use only the EOS token ID as the terminator
        eos_token_id = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=eos_token_id, # Pass a single integer for eos_token_id
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

# Example usage
model_id = "fine-tuned-medqa-llama"
summarizer_id = "facebook/bart-large-cnn"  # Example summarization model
chatbot = Chatbot(model_id, summarizer_id)

# Adding user input
chatbot.add_message("user", "Who are you?")

# Generate and print response
response = chatbot.generate_response()
print(response)

# Add more user inputs and generate responses as needed
# chatbot.add_message("user", "Additional query here.")
# response = chatbot.generate_response()
# print(response)
