from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from huggingface_hub import snapshot_download
from optimum.intel import OVModelForFeatureExtraction, OVModelForSequenceClassification
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging


MODEL_DIR = Path("models")
EMBED_MODEL_ID = "OpenVINO/bge-base-en-v1.5-int8-ov"
RERANKER_MODEL_ID = os.environ.get("RERANKER_MODEL_ID", "BAAI/bge-reranker-v2-m3")
RERANKER_OPENVINO_DIR = Path(
    os.environ.get(
        "RERANKER_OPENVINO_DIR",
        str(MODEL_DIR / f"{RERANKER_MODEL_ID.split('/')[-1]}-ov"),
    )
)
OPENVINO_DEVICE = os.environ.get("OPENVINO_DEVICE", "GPU")

transformers_logging.set_verbosity_error()


def ensure_local(repo_id: str) -> Path:
    candidate = Path(repo_id)
    if candidate.exists():
        return candidate

    local_dir = MODEL_DIR / repo_id.split("/")[-1]
    if not local_dir.exists():
        print(f"Downloading {repo_id} -> {local_dir} ...")
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    return local_dir


def ensure_converted_reranker(model_id: str) -> Path:
    if RERANKER_OPENVINO_DIR.exists():
        return RERANKER_OPENVINO_DIR

    raise FileNotFoundError(
        f"Converted reranker not found for {model_id}: {RERANKER_OPENVINO_DIR}\n"
        "Run: python convert_bge_reranker_v2_m3.py"
    )


NUM_ITERATIONS = 5

QUESTION = "What is the minimum wages order for 2022 in Malaysia?"

# Candidate passages sampled from the text you shared.
PASSAGES = [
    (
        "P.U. (A) 1406 defines the Minimum Wages Order 2022 under Act 732. "
        "It starts on 1 May 2022, with paragraph 5 from 1 Jan 2023 and "
        "paragraph 6 effective for 1 May 2022 to 31 Dec 2022."
    ),
    (
        "P.U. (A) 1407 sets minimum wages from 1 May 2022 at RM1,500 monthly, "
        "with daily and hourly rates listed. It applies to employers with five "
        "or more employees and professional activity employers under MASCO."
    ),
    (
        "P.U. (A) 1408 and 1409 describe rates for employers with fewer than "
        "five employees and area-based rates for 1 May 2022 to 31 Dec 2022: "
        "RM1,200 in city/municipal council areas and RM1,100 outside."
    ),
]


@dataclass
class ScoreRow:
    passage: str
    embed_score: float
    reranker_score: float


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def load_embedding_model(model_id: str):
    local_path = ensure_local(model_id)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = OVModelForFeatureExtraction.from_pretrained(local_path, device=OPENVINO_DEVICE)
    return tokenizer, model


def load_reranker_model(model_id: str):
    local_path = ensure_converted_reranker(model_id)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = OVModelForSequenceClassification.from_pretrained(local_path, device=OPENVINO_DEVICE)
    return tokenizer, model


def embedding_scores(tokenizer, model, question: str, passages: List[str]) -> List[float]:
    # BGE retrieval format: prepend instruction for the query side.
    texts = [f"Represent this sentence for searching relevant passages: {question}"] + passages
    batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    output = model(**batch)

    vectors = torch.nn.functional.normalize(mean_pool(output.last_hidden_state, batch["attention_mask"]), p=2, dim=1)
    return (vectors[0:1] @ vectors[1:].T).squeeze(0).tolist()


def reranker_scores(tokenizer, model, question: str, passages: List[str]) -> List[float]:
    queries = [question] * len(passages)
    batch = tokenizer(queries, passages, padding=True, truncation=True, max_length=512, return_tensors="pt")

    logits = model(**batch).logits

    # Handle both [N,1] and [N,2] heads.
    if logits.ndim == 2 and logits.shape[1] == 2:
        raw = logits[:, 1]
    elif logits.ndim == 2 and logits.shape[1] == 1:
        raw = logits[:, 0]
    else:
        raw = logits.squeeze()

    # Convert raw logits to 0..1 for easier inspection.
    return torch.sigmoid(raw).tolist()


def main() -> None:
    print(f"Embedding model: {EMBED_MODEL_ID}")
    print(f"Reranker HF model: {RERANKER_MODEL_ID}")
    print(f"Reranker OpenVINO dir: {RERANKER_OPENVINO_DIR}")
    print(f"OpenVINO device: {OPENVINO_DEVICE}")
    embed_tokenizer, embed_model = load_embedding_model(EMBED_MODEL_ID)
    reranker_tokenizer, reranker_model = load_reranker_model(RERANKER_MODEL_ID)

    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"=== Iteration {iteration}/{NUM_ITERATIONS} ===")

        embed = embedding_scores(embed_tokenizer, embed_model, QUESTION, PASSAGES)
        rerank = reranker_scores(reranker_tokenizer, reranker_model, QUESTION, PASSAGES)

        rows = [
            ScoreRow(passage=p, embed_score=e, reranker_score=r)
            for p, e, r in zip(PASSAGES, embed, rerank)
        ]

        # Combined sort: prioritize reranker signal, then embedding similarity.
        rows.sort(key=lambda x: (x.reranker_score, x.embed_score), reverse=True)

        print(f"Question: {QUESTION}\n")
        for i, row in enumerate(rows, start=1):
            print(f"[{i}] embed_cos={row.embed_score:.4f}  reranker_sigmoid={row.reranker_score:.4f}")
            print(row.passage)
            print()


if __name__ == "__main__":
    main()
