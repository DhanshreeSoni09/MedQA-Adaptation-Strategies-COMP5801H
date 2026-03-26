# Empirical Evaluation of Prompting, RAG, and QLoRA for Medical Question Answering

**Course:** COMP5801H — Generative AI and Large Language Models  
**Institution:** Carleton University, Winter 2026  
**Author:** Dhanshree Kunjalkumar Soni — 101384658  
**Project Option:** Option A — Empirical Evaluation  

---

## Overview

This project empirically compares five adaptation strategies for domain-specific medical question answering using Mistral-7B-Instruct-v0.2 as a common base model. All experiments are conducted on the MedQA (USMLE) benchmark, a dataset of four-choice questions from the United States Medical Licensing Examination.

The central research question is: under equal compute and data budgets, which adaptation strategy provides the best accuracy-efficiency trade-off for domain-specific medical question answering?

---

## Techniques Evaluated

| Technique | Description |
|---|---|
| Zero-Shot | Direct question with no examples |
| Few-Shot | 3 in-context training examples prepended |
| Chain-of-Thought | Step-by-step reasoning before answer |
| RAG k=1,3,5 | Top-k similar training examples retrieved via FAISS |
| QLoRA | 4-bit quantized LoRA fine-tuning on 2000 training examples |

---

## Key Results

| Technique | Accuracy | Macro-F1 | BERTScore F1 | Avg Latency |
|---|---|---|---|---|
| Zero-Shot | 36.60% | 33.26% | 58.01 | 9005ms |
| Few-Shot | 32.00% | 23.85% | 58.42 | 15626ms |
| CoT | 32.50% | 25.84% | 56.09 | 25265ms |
| RAG k=1 | 38.38% | 35.70% | 58.10 | 11555ms |
| RAG k=3 | 34.34% | 27.51% | 58.08 | 14881ms |
| RAG k=5 | 34.18% | 28.87% | 58.00 | 18833ms |
| QLoRA | 40.00% | 39.51% | 62.87 | 3716ms |

QLoRA achieves the best accuracy, Macro-F1, BERTScore, and lowest inference latency among all techniques evaluated.

---

## Repository Structure

    MedQA-Adaptation-Strategies-COMP5801H/
    ├── notebooks/
    │   ├── experiments.ipynb   — Sections 1 to 6, all experiments and training
    │   └── analysis.ipynb      — Sections 7 and 8, results analysis and conclusion
    ├── results/
    │   ├── zero_shot_results.csv
    │   ├── few_shot_results.csv
    │   ├── cot_results.csv
    │   ├── rag_k1_results.csv
    │   ├── rag_k3_results.csv
    │   ├── rag_k5_results.csv
    │   └── qlora_results.csv
    └── README.md

---

## Notebook Structure

Due to Kaggle's 12-hour session timeout, the project is split across two notebooks. The experiments notebook covers Sections 1 through 6 and contains all experiment outputs from the full overnight run. The analysis notebook covers Sections 7 and 8 and loads results from the saved CSV files to produce the full evaluation, cost analysis, visualizations, and error analysis.

---

## Setup and Dataset

**Base model:** mistralai/Mistral-7B-Instruct-v0.2  
**Dataset:** GBaker/MedQA-USMLE-4-options (HuggingFace)  
**Evaluation subset:** 200 questions sampled with random seed 42  
**Platform:** Kaggle (Tesla T4 x2 GPU)  

---

## Dependencies

    transformers accelerate bitsandbytes
    datasets evaluate
    bert-score rouge-score
    faiss-cpu sentence-transformers
    peft trl
    pandas numpy matplotlib seaborn scikit-learn tqdm
