# Medical Chatbot

## This repository contains the code and instructions to set up and run a medical chatbot that utilizes both a Language Model (LLM) and a Visual Question Answering (VQA) model. 

The chatbot aims to assist users by providing formal medical responses and interpreting medical images.

Project Structure
README.md: This file.
tokenize_dataset.py: Script to tokenize the dataset.
train_llm.py: Script to fine-tune the LLM.
evaluate_llm.py: Script to evaluate the LLM.
train_vqa.py: Script to fine-tune the VQA model.
evaluate_vqa.py: Script to evaluate the VQA model.

Prerequisites
Ensure you have the following Python packages installed:
pip install transformers torch datasets

Dataset
The MedQA Reasoning dataset is used for training the LLM. Make sure to download it using the datasets library.

First:

**pip install transformers torch datasets**

After that you clone this repo with the command:

**git clone CodeSailor411/AI4U_Med_Chatbot**
