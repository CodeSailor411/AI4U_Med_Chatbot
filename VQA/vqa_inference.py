import argparse
import torch
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

def parse_args():
    parser = argparse.ArgumentParser(description="VQA Inference with a VQA model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the VQA model.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image for VQA.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize VLChatProcessor and model
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    # Load the VQA model
    vl_gpt = MultiModalityCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # Define the conversation
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>Describe this MRI brain image.",
            "images": [args.image_path]
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    # Load the image and prepare inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # Run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    # Decode and print the response
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print(f"Response: {answer}")

if __name__ == "__main__":
    main()
