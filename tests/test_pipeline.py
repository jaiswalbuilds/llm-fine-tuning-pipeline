"""
Pipeline Test Suite for Dataset, Configuration, Evaluation, and Inference API
"""

import json
import os
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset_prep import validate_and_filter_item, format_to_chatml
from src.eval_bench import evaluate_json_schema, llm_judge_score
from fastapi.testclient import TestClient
from src.inference_server import app

client = TestClient(app)


def test_format_to_chatml():
    item = format_to_chatml("Generate OpenAPI spec", "API Context", '{"status": "ok"}')
    assert len(item["messages"]) == 3
    assert item["messages"][0]["role"] == "system"
    assert item["messages"][1]["role"] == "user"
    assert item["messages"][2]["role"] == "assistant"
    assert "OpenAPI" in item["messages"][1]["content"]


def test_validate_and_filter_item():
    valid_item = format_to_chatml("Short task", "", "Response")
    assert validate_and_filter_item(valid_item, max_length=500) == True

    invalid_item = {"messages": []}
    assert validate_and_filter_item(invalid_item) == False


def test_evaluate_json_schema():
    valid_json_str = '{"path": "/v1/health", "status": 200}'
    invalid_json_str = 'This is plain text with no json'
    
    assert evaluate_json_schema(valid_json_str) == True
    assert evaluate_json_schema(invalid_json_str) == False


def test_llm_judge_score():
    score = llm_judge_score("User prompt", '{"res": "ok"}', '{"res": "ok"}')
    assert score >= 7.0
    assert score <= 10.0


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_completions_endpoint():
    payload = {
        "messages": [
            {"role": "user", "content": "Generate OpenAPI spec"}
        ],
        "temperature": 0.2
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "assistant" == data["choices"][0]["message"]["role"]
