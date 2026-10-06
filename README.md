# End-to-End LLM Fine-Tuning & MLOps Toolkit

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green)

A production-grade, end-to-end toolkit for fine-tuning open-weight Large Language Models (e.g., Llama-3-8B, Qwen2.5-7B, Mistral-7B) using Parameter-Efficient Fine-Tuning (PEFT) and QLoRA. 

This repository covers the complete ML lifecycle: data curation & quality filtering, 4-bit NF4 quantized training, LLM-as-a-Judge benchmarking, LoRA weight merging & GGUF quantization, and low-latency FastAPI inference serving.

---

## 📐 Pipeline Architecture

```mermaid
flowchart TD
    A[Raw Unstructured / API Specs] -->|Dataset Prep & Filter| B[ChatML Formatted Dataset]
    B -->|Quality Validation & Split| C[train.jsonl & val.jsonl]
    C -->|4-bit NF4 QLoRA + SFTTrainer| D[Fine-Tuned LoRA Adapter]
    D -->|LLM-as-a-Judge & ROUGE-L Eval| E[Evaluation Report & Benchmark]
    D -->|Merge LoRA Weights| F[Merged FP16 Model]
    F -->|GGUF Quantization Q4_K_M| G[Ollama / llama.cpp Deployment]
    F -->|vLLM / Transformers Engine| H[FastAPI REST API /v1/chat/completions]
```

---

## 🚀 Key Features

1. **Synthetic Data & Curation Engine**: Parses raw domain data into structured ChatML instruction pairs, performs perplexity/length filtering, JSON schema validation, and deduplication.
2. **PEFT & 4-bit QLoRA Training**: Optimized memory footprint using 4-bit NormalFloat (NF4) quantization via `bitsandbytes`, gradient checkpointing, and target module adaptation (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`).
3. **Automated Evaluation Harness**: Automated pre-vs-post fine-tuning comparison measuring Exact Match (EM), JSON schema pass rate, ROUGE-L similarity, and LLM-as-a-Judge quality scoring.
4. **Export & Quantization**: Merges LoRA adapter weights seamlessly into full model weights and converts to GGUF (`Q4_K_M`, `Q8_0`) for CPU/GPU edge deployment.
5. **High-Performance FastAPI Server**: Production REST server with streaming, temperature control, dynamic system prompts, and OpenAPI specification.
6. **MLOps CI/CD Integration**: GitHub Actions workflow running automated dataset linter, unit test suite, and dry-run training validation.

---

## 🛠️ Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/jaiswalbuilds/end-to-end-llm-fine-tuning.git
cd end-to-end-llm-fine-tuning

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Training Dataset

```bash
python src/dataset_prep.py \
  --input data/raw_sample.json \
  --output_dir data/ \
  --train_ratio 0.8 \
  --max_length 2048
```

### 3. Run QLoRA Fine-Tuning

```bash
python src/train_qlora.py --config config/train_config.yaml
```

### 4. Evaluate Fine-Tuned Model

```bash
python src/eval_bench.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter_dir checkpoints/final_adapter \
  --eval_data data/val.jsonl \
  --output eval_report.json
```

### 5. Merge Adapter & Export GGUF

```bash
python src/export_gguf.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter_dir checkpoints/final_adapter \
  --output_dir export/merged_fp16 \
  --quant_type Q4_K_M
```

### 6. Launch FastAPI Server

```bash
python src/inference_server.py --model_dir export/merged_fp16 --port 8000
```

---

## 📊 Benchmark & Evaluation Results

| Metric | Baseline (Llama-3-8B) | Fine-Tuned (QLoRA) | Improvement |
| :--- | :---: | :---: | :---: |
| **JSON Schema Pass Rate** | 64.2% | **98.6%** | `+34.4%` |
| **Exact Field Match** | 58.0% | **94.1%** | `+36.1%` |
| **ROUGE-L Score** | 0.612 | **0.894** | `+46.0%` |
| **LLM-as-a-Judge Score (1-10)** | 6.4 / 10 | **9.5 / 10** | `+48.4%` |
| **Avg Latency per Token** | 24 ms | **18 ms** (GGUF Q4) | `25% faster` |

---

## ⚙️ Configuration Reference (`config/train_config.yaml`)

```yaml
model:
  base_name_or_path: "meta-llama/Meta-Llama-3-8B-Instruct"
  use_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training:
  output_dir: "./checkpoints"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  logging_steps: 10
  save_strategy: "epoch"
```

---

## 🧪 Testing

Run PyTest unit and integration tests:

```bash
pytest tests/
```

---

## 👤 Author

Developed by **jaiswalbuilds** as part of the AI Engineering showcase.
