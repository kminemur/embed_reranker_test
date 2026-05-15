from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import requests

from minimal_ovms_reranker_v2_m3 import (
    DOCUMENTS,
    GRPC_PORT,
    LOG_DIR,
    MODEL_NAME,
    NUM_ITERATIONS,
    OVMS_EXE,
    QUESTION,
    REST_PORT,
    TARGET_DEVICE,
    ovms_env,
)


MODEL_DIRS = {
    "fp32": Path("models") / "bge-reranker-v2-m3-ov",
    "int8": Path("models") / "bge-reranker-v2-m3-int8-ov",
}

TEST_CASES = [
    {
        "name": "minimum wages",
        "query": QUESTION,
        "documents": DOCUMENTS,
    },
    {
        "name": "Mercy Baker education",
        "query": " where did Mercy Baker study?",
        "documents": DOCUMENTS,
    },
]


def ensure_graph(model_dir: Path) -> None:
    graph_text = f'''input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {{
    name: "{MODEL_NAME}"
    calculator: "RerankCalculatorOV"
    input_side_packet: "RERANK_NODE_RESOURCES:rerank_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {{
        [type.googleapis.com / mediapipe.RerankCalculatorOVOptions]: {{
            models_path: "./",
            max_allowed_chunks: 10000,
            target_device: "{TARGET_DEVICE}",
            plugin_config: '{{"NUM_STREAMS":"1"}}',
        }}
    }}
}}
'''
    (model_dir / "graph.pbtxt").write_text(graph_text, encoding="utf-8")


def start_ovms(model_dir: Path) -> subprocess.Popen:
    LOG_DIR.mkdir(exist_ok=True)
    log_file = (LOG_DIR / "minimal_ovms_reranker_v2_m3_different_prompt.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            str(OVMS_EXE.resolve()),
            "--model_path",
            str(model_dir.resolve()),
            "--model_name",
            MODEL_NAME,
            "--rest_port",
            str(REST_PORT),
            "--port",
            str(GRPC_PORT),
        ],
        cwd=str(OVMS_EXE.parent.resolve()),
        env=ovms_env(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    url = f"http://localhost:{REST_PORT}/v1/models/{MODEL_NAME}"
    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError(f"OVMS exited early with code {proc.returncode}")
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            time.sleep(1)

    proc.terminate()
    raise TimeoutError("OVMS did not start in time")


def rerank(query: str, documents: list[str]) -> dict:
    response = requests.post(
        f"http://localhost:{REST_PORT}/v3/rerank",
        json={"model": MODEL_NAME, "query": query, "documents": documents},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bge-reranker-v2-m3 OVMS tests with different prompts."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_DIRS),
        default="int8",
        help="Model precision to use. Default: int8",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Custom OpenVINO model directory. Overrides --model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir or MODEL_DIRS[args.model]
    model_label = "custom" if args.model_dir else args.model

    print(f"Using model: {MODEL_NAME}")
    print(f"Model precision: {model_label}")
    print(f"Model directory: {model_dir}")
    print(f"Target device: {TARGET_DEVICE}")
    print()

    ensure_graph(model_dir)
    proc = start_ovms(model_dir)
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
