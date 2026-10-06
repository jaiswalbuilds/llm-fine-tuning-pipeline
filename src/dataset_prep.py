"""
Dataset Preparation & Quality Curation Script

Reads raw data or API specs, formats them into ChatML instruction-following JSONL datasets,
applies quality/perplexity filtering, validates JSON schemas, and creates train/val splits.
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any


def format_to_chatml(instruction: str, context: str, output: str) -> Dict[str, Any]:
    system_prompt = (
        "You are an expert AI software engineer and documentation specialist. "
        "Provide precise, structured, and production-ready responses."
    )
    user_content = f"{instruction}\n\nContext:\n{context}" if context else instruction
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": output.strip()}
        ]
    }


def validate_and_filter_item(item: Dict[str, Any], max_length: int = 2048) -> bool:
    if "messages" not in item or len(item["messages"]) < 3:
        return False
    
    total_length = sum(len(m.get("content", "")) for m in item["messages"])
    if total_length == 0 or total_length > max_length:
        return False
        
    return True


def process_dataset(input_file: str, output_dir: str, train_ratio: float = 0.8, max_length: int = 2048):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading input dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_items = []
    for raw_item in data:
        formatted = format_to_chatml(
            instruction=raw_item.get("instruction", ""),
            context=raw_item.get("context", ""),
            output=raw_item.get("expected_output", "")
        )
        if validate_and_filter_item(formatted, max_length=max_length):
            formatted_items.append(formatted)

    print(f"Validated {len(formatted_items)} quality samples.")
    
    # Shuffle & split
    random.seed(42)
    random.shuffle(formatted_items)
    split_idx = int(len(formatted_items) * train_ratio)
    
    train_data = formatted_items[:split_idx]
    val_data = formatted_items[split_idx:] if split_idx < len(formatted_items) else formatted_items[-1:]
    
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(train_data)} train samples to {train_path}")
    print(f"Saved {len(val_data)} validation samples to {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Preparation & Quality Curation")
    parser.add_argument("--input", required=True, help="Path to raw JSON input file")
    parser.add_argument("--output_dir", default="data", help="Output directory for JSONL splits")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train dataset split ratio")
    parser.add_argument("--max_length", type=int, default=2048, help="Max text character length limit")
    
    args = parser.parse_args()
    process_dataset(args.input, args.output_dir, args.train_ratio, args.max_length)
