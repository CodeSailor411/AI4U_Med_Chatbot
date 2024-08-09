import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoFeatureExtractor
from PIL import Image

def load_model_and_tokenizer(model_path, tokenizer_path, feature_extractor_path):
    # Load the fine-tuned model, tokenizer, and feature extractor
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)
    return model, tokenizer, feature_extractor

def perform_inference(model, tokenizer, feature_extractor, image_path, question):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Tokenize the question
    question_encoding = tokenizer(question, return_tensors="pt")
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=inputs['pixel_values'],
            input_ids=question_encoding['input_ids'],
            attention_mask=question_encoding['attention_mask'],
            max_new_tokens=512,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode and return the response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    # Define model, tokenizer, feature extractor paths and input parameters
    model_path = "./fine-tuned-vqa-model"
    tokenizer_path = "openbmb/MiniCPM-Llama3-V-2_5"
    feature_extractor_path = "openbmb/MiniCPM-Llama3-V-2_5"
    image_path = "./brain_mri_image.png"
    question = "What can you describe from this MRI image?"

    # Load model, tokenizer, and feature extractor
    model, tokenizer, feature_extractor = load_model_and_tokenizer(
        model_path,
        tokenizer_path,
        feature_extractor_path
    )
    
    # Perform inference
    answer = perform_inference(
        model,
        tokenizer,
        feature_extractor,
        image_path,
        question
    )
    
    # Print the result
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
