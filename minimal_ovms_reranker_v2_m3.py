from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import requests


OVMS_EXE = Path("ovms_2026.1.0") / "ovms" / "ovms.exe"
MODEL_DIR = Path("models") / "bge-reranker-v2-m3-ov"
MODEL_NAME = "bge-reranker-v2-m3"
REST_PORT = 9001
GRPC_PORT = 9101
TARGET_DEVICE = os.environ.get("OVMS_TARGET_DEVICE", "GPU")
LOG_DIR = Path("logs")

QUESTION = "What is the minimum wages order for 2022 in Malaysia?"
DOCUMENTS = [
    "P.U. (A) 1406 defines the Minimum Wages Order 2022 under Act 732.",
    "P.U. (A) 1407 sets minimum wages from 1 May 2022 at RM1,500 monthly.",
    "P.U. (A) 1408 and 1409 describe rates for employers with fewer than five employees.",
]
NUM_ITERATIONS = 5


def ensure_graph() -> None:
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
    (MODEL_DIR / "graph.pbtxt").write_text(graph_text, encoding="utf-8")


def ovms_env() -> dict[str, str]:
    ovms_dir = OVMS_EXE.parent.resolve()
    python_home = ovms_dir / "python"
    env = os.environ.copy()
    env["OVMS_DIR"] = str(ovms_dir)
    env["PYTHONHOME"] = str(python_home)
    env["PATH"] = os.pathsep.join([str(ovms_dir), str(python_home), str(python_home / "Scripts")]) + os.pathsep + env["PATH"]
    return env


def start_ovms() -> subprocess.Popen:
    LOG_DIR.mkdir(exist_ok=True)
    log_file = (LOG_DIR / "minimal_ovms_reranker_v2_m3.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            str(OVMS_EXE.resolve()),
            "--model_path",
            str(MODEL_DIR.resolve()),
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


def rerank() -> dict:
    response = requests.post(
        f"http://localhost:{REST_PORT}/v3/rerank",
        json={"model": MODEL_NAME, "query": QUESTION, "documents": DOCUMENTS},
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
            result = rerank()
            for row in result["results"]:
                print(f'{row["index"]}: {row["relevance_score"]:.4f}  {DOCUMENTS[row["index"]]}')
            print()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
