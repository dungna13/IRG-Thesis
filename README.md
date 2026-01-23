# Qwen-IRG: Visual Reasoning and Refinement for Image Generation

## Project Overview

This project, developed as part of a graduation thesis, presents a comprehensive pipeline for enhancing image generation capabilities through the fine-tuning of Large Language Models (LLMs). The system, titled Qwen-IRG (Instruction-Reasoning-Generation), leverages the Qwen model to provide detailed visual reasoning, prompt refinement, and quality diagnostics for text-to-image generation tasks.

By integrating CLIP feature awareness and multi-iteration reasoning, the model helps bridge the gap between user intent and generated image quality, offering specific guidance on composition, lighting, style, and technical attributes.

## Key Features

1.  **Synthetic Dataset Generation**: A robust system for generating high-quality training data, including:
    *   Initial visual reasoning steps.
    *   Multi-iteration refinement sequences.
    *   CLIP-aware feature interpretation (analyzing brightness, contrast, highlights).
    *   Problem-solving scenarios (diagnosing under/overexposure, low contrast).

2.  **Fine-tuned Qwen Model**: A specialized version of Qwen (e.g., Qwen 2.5) fine-tuned on the generated dataset using LoRA (Low-Rank Adaptation) and QLoRA techniques. This model acts as an expert visual reasoning assistant.

3.  **Visual Reasoning Capabilities**:
    *   **Composition & Lighting Analysis**: Detailed breakdown of scene structure and illumination.
    *   **Iterative Refinement**: Step-by-step guidance to improve image fidelity.
    *   **Quality Diagnostics**: Automated detection and correction of common image generation issues.

## Repository Structure

*   **Workflow-CODE**
    *   `irg-1-dataset-generation.ipynb`: Jupyter notebook for Phase 1. Handles the generation of the comprehensive training dataset, simulating CLIP features and creating reasoning chains.
    *   `irg-2-qwen-finetuning.ipynb`: Jupyter notebook for Phase 2. Implements the fine-tuning pipeline for Qwen using PEFT/LoRA, including data preparation and training loop configuration.
    *   `irg-imagegeneration.ipynb`: Notebook for Phase 3. Demonstrates the image generation process using the fine-tuned model's guidance.
    *   `phase3-benchmark.ipynb`: Benchmarking suite to evaluate the model's performance and improvement metrics.
    *   `prompts.txt`: A collection of standardized prompts used for testing and benchmarking.

*   **Final_check_2.pdf**: The complete thesis report documentation.

## Installation and Requirements

### Prerequisites

*   Python 3.8 or higher
*   CUDA-compatible GPU (Recommended: NVIDIA T4 or better for fine-tuning)

### Dependencies

The project relies on the following key libraries:

*   PyTorch
*   Transformers (Hugging Face)
*   PEFT (Parameter-Efficient Fine-Tuning)
*   Accelerate
*   Datasets
*   Pandas & NumPy

To install the necessary packages, you can run:

    pip install torch transformers peft accelerate datasets pandas numpy

## Usage

### 1. Dataset Generation

Run the `irg-1-dataset-generation.ipynb` notebook to create the training dataset. This script allows you to configure the number of sequences and refinement examples. The output will be a JSON/CSV file containing the structured training data.

### 2. Model Fine-tuning

Execute `irg-2-qwen-finetuning.ipynb` to train the Qwen model. Ensure you have the base Qwen model available (e.g., via Hugging Face/Kaggle). The script supports QLoRA for memory-efficient training on consumer GPUs.

### 3. Inference and Generation

Use the `irg-imagegeneration.ipynb` notebook to load the fine-tuned adapter and generate visual reasoning for your prompts. The model will accept a raw text prompt and output a detailed technical description or refinement plan to guide your image generation tool (e.g., Stable Diffusion).

## Methodology

The core innovation of this thesis lies in the "Feature-Aware Refinement" loop. Unlike standard prompt engineering, this system simulates a feedback loop where the LLM analyzes theoretical image features (derived from CLIP statistics) to propose concrete corrective actions, mimicking the workflow of a professional photographer or digital artist.

## Acknowledgments

This research makes use of the Qwen language model series by Alibaba Cloud and the PEFT library by Hugging Face. Special thanks to the open-source community for the tools enabling efficient LLM fine-tuning.

## License

Copyright (c) 2025 Ngô Anh Dũng. This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. You may not use this work for commercial purposes.
