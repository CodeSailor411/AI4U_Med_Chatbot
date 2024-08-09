# Medical Chatbot

## This repository contains the code and instructions to set up and run a medical chatbot that utilizes both a Language Model (LLM) and a Visual Question Answering (VQA) model. 

The chatbot aims to assist users by providing formal medical responses and interpreting medical images.

Project Structure:
README.md: This file.
tokenize_dataset.py: Script to tokenize the dataset.
train_llm.py: Script to fine-tune the LLM.
evaluate_llm.py: Script to evaluate the LLM.
train_vqa.py: Script to fine-tune the VQA model.
evaluate_vqa.py: Script to evaluate the VQA model.


Dataset:

The MedQA  dataset is used for training the LLM. Make sure to download it using the datasets library.

Preparing environment:
**python -m venv med_chatbot**
Then:
**med_chatbot\Scripts\activate**
Prerequisites:

Ensure you have the following Python packages installed:


First run this command:

**pip install transformers torch datasets huggingface_hub deepseek-vl**

After that you clone this repo with the command:

**git clone CodeSailor411/AI4U_Med_Chatbot**

Then:
**cd AI4U_Med_Chatbot/LLM**

or:

**cd AI4U_Med_Chatbot/VQA**


For cli_inference run :

**python vqa_inference_cli.py --model_path deepseek-ai/deepseek-vl-7b-chat --image_path ./brain_mri_image.png**

