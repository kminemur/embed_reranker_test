from __future__ import annotations

import requests

from minimal_ovms_reranker_v2_m3 import (
    DOCUMENTS,
    MODEL_NAME,
    NUM_ITERATIONS,
    QUESTION,
    REST_PORT,
    ensure_graph,
    start_ovms,
)


TEST_CASES = [
    {
        "name": "minimum wages",
        "query": QUESTION,
        "documents": DOCUMENTS,
    },
    {
        "name": "Mercy Baker education",
        "query": " where did Mercy Baker study?",
        "documents": [
            "Mercy Baker studied at the University of Nairobi before beginning her public health career.",
            "Mercy Baker later worked with community clinics and maternal health programs.",
            "Baker Mercy Hospital was founded in 1912 and is not connected to Mercy Baker.",
        ],
    },
]


def rerank(query: str, documents: list[str]) -> dict:
    response = requests.post(
        f"http://localhost:{REST_PORT}/v3/rerank",
        json={"model": MODEL_NAME, "query": query, "documents": documents},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    ensure_graph()
    proc = start_ovms()
    try:
        for iteration in range(1, NUM_ITERATIONS + 1):
            print(f"=== Iteration {iteration}/{NUM_ITERATIONS} ===")
            for case in TEST_CASES:
                print(f'--- {case["name"]} ---')
                print(f'Prompt: {case["query"]}')

                documents = case["documents"]
                result = rerank(case["query"], documents)
                for row in result["results"]:
                    print(f'{row["index"]}: {row["relevance_score"]:.4f}  {documents[row["index"]]}')
                print()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
