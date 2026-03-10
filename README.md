# Gemini-IRG: Visual Reasoning and Autonomous Refinement for Image Generation

## Project Overview

This project, transitioning from its V1.0 thesis implementation to a V2.0 architecture, presents a comprehensive pipeline for enhancing image generation capabilities through the reasoning power of Large Language Models (LLMs). The system, titled Gemini-IRG (Instruction-Reasoning-Generation), leverages the Gemini LLM to provide detailed visual reasoning, prompt refinement, and quality diagnostics for text-to-image generation tasks executed via Stability AI (Stable Diffusion XL).

By integrating CLIP feature awareness, Autonomous Multi-Agent reasoning, and Retrieval-Augmented Generation (RAG), the model bridges the gap between user intent and generated image quality, offering specific, self-correcting guidance on composition, lighting, style, and technical attributes without human intervention.

## V2.0 Major Leap: Autonomous Multi-Agent & RAG Architecture

To push the boundaries of traditional prompt engineering, the system has been fundamentally re-architected into a closed-loop autonomous system:

* Multi-Agent Workflow: Moved away from zero-shot generation to an iterative, self-correcting pipeline. The workflow automatically evaluates image statistics and feeds them to an Expert Gemini Agent for dynamic adjustment (e.g., boosting contrast if std < 0.12).
* Retrieval-Augmented Generation (RAG): Injects high-quality historical refinement cases as few-shot context to the LLM. Before generating a prompt, the system queries past successful cases to guide the Expert Agent's reasoning format and decision-making.
* Structured Heuristic Diagnostics: Implemented strict Regex parsing to force the LLM to output predictable formats containing precise actionable steps (Diagnosis -> Actions -> Refined Prompt).

## Key Features

1. Iterative Self-Correction Pipeline: A closed-loop system where the LLM acts as an art director, analyzing generated image statistics and autonomously issuing corrective prompts over multiple iterations.
2. Gemini-Powered Reasoning: Utilizing Gemini as an expert visual reasoning assistant with a deep understanding of structural layouts and prompt composition.
3. RAG-Powered Context: Dynamic injection of historical prompt-refinement data to stabilize LLM outputs and improve reasoning consistency.
4. Feature-Aware Feedback: Using statistical descriptors derived from images to guide the LLM's understanding of the current generation.

## Empirical Findings (Thesis Data)

The system's modular reasoning approach has been rigorously evaluated against standard single-shot generation (Base SDXL). Key findings include:
* Compositional Accuracy: The 2-iteration reasoning loop achieves a +7.74% improvement in compositional accuracy (from 0.3497 to 0.3768) compared to the zero-shot baseline.
* Aesthetic Improvement: The refinement process yields monotonic improvements in visual quality, increasing the aesthetic score by up to +3.08% over the base model.
* Reasoning Efficacy: Across all iteration depths, the LLM reasoning-guided pipeline consistently outperformed no-reasoning image-to-image refinement baselines in preserving semantic alignment and compositional integrity.

## Repository Structure

* src/ (V2.0 Core Architecture)
    * core/workflow.py: The orchestrator handling the Multi-Agent loop and multi-iteration reasoning.
    * agents/expert_agent.py: The Gemini LLM Agent enforcing heuristic rules for diagnostics and refinement.
    * services/rag_service.py: Handles contextual data retrieval from historical runs.
    * services/image_service.py: Evaluates and generates images using Stability AI API.
* Workflow-CODE/ (V1.0 & Data Generation)
    * irg-1-dataset-generation.ipynb: Generates the comprehensive training dataset, simulating CLIP features.
    * phase3-benchmark.ipynb: Benchmarking suite to evaluate the model's improvement metrics via GenEval.
* Final_check_2.pdf: The complete thesis report documentation containing full methodology and experimental data.

## Installation and Requirements

### Prerequisites
* Python 3.8 or higher
* Valid API Keys for Gemini and Stability AI

### Dependencies
The project relies on the following key libraries:
pip install google-generativeai requests pandas numpy pillow

## Usage

### Autonomous Multi-Agent Generation (V2.0)
Run the core workflow to experience autonomous self-correction. The IRGWorkflow will automatically retrieve RAG context, generate an initial image via Stability AI, analyze its statistics, and run iterations to refine it.

## Methodology

The core innovation of this project lies in the "Feature-Aware Refinement" loop. Unlike standard prompt engineering, this system simulates a feedback loop where an Expert LLM Agent analyzes quantitative image features to propose concrete corrective actions, mimicking the exact workflow of a professional photographer or digital artist.

## License

Copyright (c) 2025 Ngô Anh Dũng. This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. You may not use this work for commercial purposes.
