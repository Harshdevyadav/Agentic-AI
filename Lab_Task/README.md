SLM Fine-Tuning: Gemma-3-1B-it on Dolly-15k 
A professional laboratory implementation of fine-tuning a Small Language Model (SLM) using 4-bit quantization (QLoRA) for efficient instruction following. This project was developed as part of the Agentic AI Lab Task.

Project Overview
The goal of this project is to demonstrate the feasibility of fine-tuning models with less than 3B parameters on consumer-grade hardware (Google Colab T4 GPU). By utilizing the Gemma-3-1B-it model and the Databricks-Dolly-15k dataset, the model was trained to follow human-like instructions more effectively.

Tech Stack
Model: Google Gemma-3-1B-it (<3B parameters)

Dataset: Databricks-Dolly-15k (Hugging Face)

Frameworks: transformers, peft, bitsandbytes, trl

Hardware: Google Colab T4 GPU (16GB VRAM)

Getting Started
1. Requirements
To reproduce this training, you will need a Hugging Face account and a Write Access Token.

2. Installation
Run the following in your environment to install dependencies:

Bash

pip install -q -U bitsandbytes transformers peft accelerate datasets trl
3. Usage
The main implementation is contained in Fine_tuning_Lab_task1.ipynb. Open this file in Google Colab, provide your HF_TOKEN in the Secrets tab, and run all cells.

Results & Observations
Efficiency: Used 4-bit quantization to load the 1.1B parameter model, consuming only ~1.5GB of VRAM initially.

Training Stability: The training loss decreased from 2.27 to 1.88 over 50 steps, showing successful pattern adaptation.

Decoding Strategy: Implemented Sampling with a repetition_penalty=1.2 to resolve initial issues with repetitive loops in the output.

License
This project is licensed under the MIT License.
