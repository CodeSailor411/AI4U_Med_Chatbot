import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

class VQAChatbot:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda().eval()
        self.processor = VLChatProcessor.from_pretrained(model_path)

    def generate_response(self, image_path, question):
        # Prepare conversation
        conversation = [
            {
                "role": "User",
                "content": question,
                "images": [image_path]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # Load and prepare images
        pil_images = load_pil_images(conversation)
        inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.model.device)

        # Generate response
        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return response

# Example usage
model_path = "./fine-tuned-vqa-model"
tokenizer_path = model_path
vqa_chatbot = VQAChatbot(model_path, tokenizer_path)

# Test inference
image_path = "./images/training_pipelines.png"  # Replace with your image path
question = "Describe each stage of this image."
response = vqa_chatbot.generate_response(image_path, question)
print(response)
