from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from optimum.intel import OVModelForFeatureExtraction, OVModelForSequenceClassification
from transformers import AutoTokenizer


EMBED_MODEL_ID = "OpenVINO/bge-base-en-v1.5-int8-ov"
RERANKER_MODEL_ID = "OpenVINO/bge-reranker-base-int8-ov"

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
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def load_embedding_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = OVModelForFeatureExtraction.from_pretrained(model_id, device="GPU")
    return tokenizer, model


def load_reranker_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = OVModelForSequenceClassification.from_pretrained(model_id, device="GPU")
    return tokenizer, model


def embedding_scores(question: str, passages: List[str]) -> List[float]:
    tokenizer, model = load_embedding_model(EMBED_MODEL_ID)

    # BGE retrieval format: prepend instruction for the query side.
    texts = [f"Represent this sentence for searching relevant passages: {question}"] + passages
    batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        output = model(**batch)

    vectors = mean_pool(output.last_hidden_state, batch["attention_mask"])
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)

    q = vectors[0:1]
    docs = vectors[1:]
    sims = (q @ docs.T).squeeze(0).tolist()
    return [float(x) for x in sims]


def reranker_scores(question: str, passages: List[str]) -> List[float]:
    tokenizer, model = load_reranker_model(RERANKER_MODEL_ID)

    queries = [question] * len(passages)
    batch = tokenizer(queries, passages, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        logits = model(**batch).logits

    # Handle both [N,1] and [N,2] heads.
    if logits.ndim == 2 and logits.shape[1] == 2:
        raw = logits[:, 1]
    elif logits.ndim == 2 and logits.shape[1] == 1:
        raw = logits[:, 0]
    else:
        raw = logits.squeeze()

    # Convert raw logits to 0..1 for easier inspection.
    probs = torch.sigmoid(raw)
    return [float(x) for x in probs.tolist()]


def main() -> None:
    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"=== Iteration {iteration}/{NUM_ITERATIONS} ===")

        embed = embedding_scores(QUESTION, PASSAGES)
        rerank = reranker_scores(QUESTION, PASSAGES)

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
