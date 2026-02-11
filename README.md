# Semiconductor Simulation Code Generation with Generative AI

A **generative language model system for producing semiconductor device simulation code** from natural-language design specifications.  
Developed as the final project for *CSC 375 – Generative AI*, focusing on constrained model design, code generation accuracy, and generalization.

---

## Project Overview

This project explores the use of **small-scale language models (≤1B parameters)** to generate **SPICE-compatible semiconductor simulation code** for use with **Silvaco TCAD**, an industry-standard platform for semiconductor device modeling and circuit analysis.

The system takes structured or semi-structured natural-language descriptions of device designs and outputs simulation code that captures device geometry, materials, and electrical characteristics.

The primary emphasis is on **generalization, methodological rigor, and correctness**, rather than memorization of training examples.

---

## Core Objectives

- Translate natural-language device specifications into simulation code
- Operate under strict model size constraints (≤1B parameters)
- Evaluate generalization on unseen test prompts
- Demonstrate reproducible training and evaluation workflows

---

## Model & Methodology

### Model Constraints
- Maximum model size: **≤1B parameters**
- Models selected from approved open-source architectures (e.g., GPT-2 variants, T5, Qwen, LLaMA-class small models)

### Techniques Explored
- Prompt engineering for structured code generation
- Supervised fine-tuning on instruction–code pairs
- Careful preprocessing to preserve domain-specific syntax
- Evaluation on held-out, hidden test samples

The final approach prioritizes **clarity, stability, and interpretability** over architectural complexity.

---

## Dataset Notes

- Training data consists of paired **natural-language instructions and Silvaco-compatible simulation code**
- A separate **hidden test set** is used for evaluation to assess generalization
- **Dataset usage is restricted to course requirements**
  - The dataset is **not included** in this repository
  - No redistribution or public hosting of dataset files

---

## System Architecture

1. **Input Processing**  
   Natural-language specifications are normalized and formatted for model input.

2. **Code Generation**  
   The trained language model produces simulation code following domain-specific syntax constraints.

3. **Evaluation**  
   Generated code is evaluated against reference outputs using automated and qualitative checks.

---

## Training
```
python train.py
```
## Evaulation
```
python evaluate.py
```

---
## Skills & Concepts Demonstrated

### This project demonstrates applied skills in:

- Generative AI and language models
- Code generation systems
- Model fine-tuning under parameter constraints
- Prompt engineering
- Dataset preprocessing and evaluation
- Reproducible ML workflows
- Domain-specific language modeling

---

## Authors
- Christopher Anzilotti
- Alex Power
- Thomas Morrissey

Final Project for CSC 375 – Generative AI
Quinnipiac University
Instructor: Prof. Rongyu 

---

## Notes for Reviewers
This repository is structured to reflect professional ML experimentation practices while complying with course-specific constraints. Dataset files are intentionally excluded, and the focus is placed on methodology, implementation quality, and evaluation rigor.


