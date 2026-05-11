
import os
import zipfile
import requests
import subprocess
import time
import json
import shutil
from typing import List
from pathlib import Path
from huggingface_hub import snapshot_download


# === OVMS Download/Extract/Start ===
OVMS_URL_ZIP = "https://github.com/openvinotoolkit/model_server/releases/download/v2026.1/ovms_windows_2026.1.0_python_on.zip"
ZIP_NAME = "ovms_windows_2026.1.0_python_on.zip"
EXTRACT_DIR = "ovms_2026.1.0"
OVMS_DIR = Path(EXTRACT_DIR) / "ovms"
OVMS_EXE = OVMS_DIR / "ovms.exe"
LOG_DIR = Path("logs")
MODEL_BASE_DIR = Path("models")
OVMS_MODEL_ROOT = MODEL_BASE_DIR / "ovms_layout"
EMBED_MODEL_ID = "OpenVINO/bge-base-en-v1.5-int8-ov"
RERANKER_MODEL_ID = "OpenVINO/bge-reranker-base-int8-ov"
EMBED_REST_PORT = 9000
RERANKER_REST_PORT = 9001
EMBED_GRPC_PORT = 9100
RERANKER_GRPC_PORT = 9101
EMBED_MODEL_NAME = "bge"
RERANKER_MODEL_NAME = "bge-reranker"
EMBED_OVMS_URL = f"http://localhost:{EMBED_REST_PORT}/v3/embeddings"
RERANKER_OVMS_URL = f"http://localhost:{RERANKER_REST_PORT}/v3/rerank"


def ensure_model(repo_id: str) -> Path:
    local_dir = MODEL_BASE_DIR / repo_id.split("/")[-1]
    if local_dir.exists():
        print(f"Model {local_dir} already exists, skipping download.")
        return local_dir
    print(f"\n[ERROR] Model not found: {local_dir}")
    print(f"以下のいずれかの方法でモデルを配置してください:")
    print(f"  1. test_bge_outputs.py を先に実行する（自動ダウンロード）")
    print(f"  2. HuggingFace から手動でダウンロード: https://huggingface.co/{repo_id}")
    print(f"  配置先: {local_dir.absolute()}\n")
    raise SystemExit(1)

def ensure_ovms_model_layout(source_dir: Path) -> Path:
    model_root = OVMS_MODEL_ROOT / source_dir.name
    version_dir = model_root / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    for source_file in source_dir.iterdir():
        if not source_file.is_file():
            continue
        dest_file = version_dir / source_file.name
        if dest_file.exists() and dest_file.stat().st_size == source_file.stat().st_size:
            continue
        if dest_file.exists():
            dest_file.unlink()
        try:
            os.link(source_file, dest_file)
        except OSError:
            shutil.copy2(source_file, dest_file)

    print(f"OVMS model layout ready: {model_root}")
    return model_root

def ensure_ovms_graph(source_dir: Path, graph_kind: str, model_name: str) -> Path:
    graph_path = source_dir / "graph.pbtxt"
    if graph_kind == "embeddings":
        graph_text = f'''input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {{
    name: "{model_name}"
    calculator: "EmbeddingsCalculatorOV"
    input_side_packet: "EMBEDDINGS_NODE_RESOURCES:embeddings_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {{
        [type.googleapis.com / mediapipe.EmbeddingsCalculatorOVOptions]: {{
            models_path: "./",
            normalize_embeddings: true,
            truncate: false,
            pooling: CLS,
            target_device: "CPU",
            plugin_config: '{{"NUM_STREAMS":"1"}}',
        }}
    }}
}}
'''
    elif graph_kind == "rerank":
        graph_text = f'''input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {{
    name: "{model_name}"
    calculator: "RerankCalculatorOV"
    input_side_packet: "RERANK_NODE_RESOURCES:rerank_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {{
        [type.googleapis.com / mediapipe.RerankCalculatorOVOptions]: {{
            models_path: "./",
            max_allowed_chunks: 10000,
            target_device: "CPU",
            plugin_config: '{{"NUM_STREAMS":"1"}}',
        }}
    }}
}}
'''
    else:
        raise ValueError(f"Unknown OVMS graph kind: {graph_kind}")

    if not graph_path.exists() or graph_path.read_text(encoding="utf-8", errors="replace") != graph_text:
        graph_path.write_text(graph_text, encoding="utf-8")
    print(f"OVMS {graph_kind} graph ready: {graph_path}")
    return source_dir


QUESTION = "What is the minimum wages order for 2022 in Malaysia?"
PASSAGES = [
    "P.U. (A) 1406 defines the Minimum Wages Order 2022 under Act 732. It starts on 1 May 2022, with paragraph 5 from 1 Jan 2023 and paragraph 6 effective for 1 May 2022 to 31 Dec 2022.",
    "P.U. (A) 1407 sets minimum wages from 1 May 2022 at RM1,500 monthly, with daily and hourly rates listed. It applies to employers with five or more employees and professional activity employers under MASCO.",
    "P.U. (A) 1408 and 1409 describe rates for employers with fewer than five employees and area-based rates for 1 May 2022 to 31 Dec 2022: RM1,200 in city/municipal council areas and RM1,100 outside."
]


def download_ovms_zip(url: str, dest: str):
    if not os.path.exists(dest):
        print(f"\n[ERROR] {dest} が見つかりません。")
        print(f"以下のURLから手動でダウンロードし、このフォルダに配置してください:")
        print(f"  {url}")
        print(f"配置先: {os.path.abspath(dest)}\n")
        raise SystemExit(1)
    else:
        print(f"{dest} already exists, skipping download.")

def extract_zip(zip_path: str, extract_to: str):
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")
    else:
        print(f"{extract_to} already exists, skipping extraction.")

def build_ovms_env() -> dict:
    ovms_dir = OVMS_DIR.resolve()
    python_home = ovms_dir / "python"

    env = os.environ.copy()
    env["OVMS_DIR"] = str(ovms_dir)
    if python_home.exists():
        env["PYTHONHOME"] = str(python_home)
        path_entries = [ovms_dir, python_home, python_home / "Scripts"]
    else:
        path_entries = [ovms_dir]

    env["PATH"] = os.pathsep.join(str(entry) for entry in path_entries) + os.pathsep + env.get("PATH", "")
    return env

def read_log_tail(log_path: Path, max_chars: int = 4000) -> str:
    if not log_path.exists():
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]

def start_ovms(model_dir: str, model_name: str, rest_port: int, grpc_port: int):
    ovms_exe = OVMS_EXE.resolve()
    model_path = Path(model_dir).resolve()
    if not ovms_exe.exists():
        raise FileNotFoundError(f"OVMS executable not found: {OVMS_EXE}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"ovms_{model_name}_{rest_port}.log"
    cmd = [
        str(ovms_exe),
        "--model_path", str(model_path),
        "--model_name", model_name,
        "--rest_port", str(rest_port),
        "--port", str(grpc_port),
    ]
    print(f"Starting OVMS ({model_name}): {' '.join(cmd)}")
    print(f"OVMS ({model_name}) log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(OVMS_DIR.resolve()),
            env=build_ovms_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    # サーバ起動待ち
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            log_tail = read_log_tail(log_path)
            raise RuntimeError(
                f"OVMS ({model_name}) exited during startup with code {exit_code}.\n"
                f"Log tail ({log_path}):\n{log_tail}"
            )
        try:
            r = requests.get(f"http://localhost:{rest_port}/v1/models/{model_name}", timeout=1)
            if r.status_code == 200:
                print(f"OVMS ({model_name}) REST is up on port {rest_port}.")
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    log_tail = read_log_tail(log_path)
    proc.terminate()
    raise RuntimeError(
        f"OVMS ({model_name}) did not start in time.\n"
        f"Log tail ({log_path}):\n{log_tail}"
    )


# 推論リクエスト（embedding用）
def make_ovms_embedding_request(inputs: List[str]):
    payload = {
        "model": EMBED_MODEL_NAME,
        "input": inputs,
    }
    response = requests.post(EMBED_OVMS_URL, json=payload, timeout=30)
    if not response.ok:
        raise RuntimeError(f"Embedding request failed: {response.status_code}\n{response.text}")
    response.raise_for_status()
    return response.json()

# 推論リクエスト（reranker用）
def make_ovms_reranker_request(questions: List[str], passages: List[str]):
    if not questions:
        return {"results": []}
    if len(set(questions)) != 1:
        raise ValueError("OVMS /v3/rerank accepts one query per request.")
    payload = {
        "model": RERANKER_MODEL_NAME,
        "query": questions[0],
        "documents": passages,
    }
    response = requests.post(RERANKER_OVMS_URL, json=payload, timeout=30)
    if not response.ok:
        raise RuntimeError(f"Reranker request failed: {response.status_code}\n{response.text}")
    response.raise_for_status()
    return response.json()


from dataclasses import dataclass

NUM_ITERATIONS = 5

@dataclass
class ScoreRow:
    passage: str
    embed_score: float
    reranker_score: float

def main():
    # 1. OVMSダウンロード・展開
    download_ovms_zip(OVMS_URL_ZIP, ZIP_NAME)
    extract_zip(ZIP_NAME, EXTRACT_DIR)

    # 2. モデル自動ダウンロード
    embed_model_dir = ensure_ovms_graph(ensure_model(EMBED_MODEL_ID), "embeddings", EMBED_MODEL_NAME)
    reranker_model_dir = ensure_ovms_graph(ensure_model(RERANKER_MODEL_ID), "rerank", RERANKER_MODEL_NAME)

    # 3. OVMS起動（embeddingとrerankerを別ポートで起動）
    embed_proc = start_ovms(str(embed_model_dir), EMBED_MODEL_NAME, EMBED_REST_PORT, EMBED_GRPC_PORT)
    reranker_proc = start_ovms(str(reranker_model_dir), RERANKER_MODEL_NAME, RERANKER_REST_PORT, RERANKER_GRPC_PORT)

    try:
        for iteration in range(1, NUM_ITERATIONS + 1):
            print(f"=== Iteration {iteration}/{NUM_ITERATIONS} ===")

            # Embeddingスコア取得
            query = f"Represent this sentence for searching relevant passages: {QUESTION}"
            texts = [query] + PASSAGES
            embed_result = make_ovms_embedding_request(texts)
            # 結果からベクトルを取得
            vectors = [item["embedding"] for item in embed_result["data"]]  # [query, doc1, doc2, doc3]
            import numpy as np
            vectors = np.array(vectors).reshape(len(texts), -1)
            q = vectors[0:1]
            docs = vectors[1:]
            sims = (q @ docs.T) / (np.linalg.norm(q, axis=1, keepdims=True) * np.linalg.norm(docs, axis=1))
            embed_scores = sims.flatten().tolist()

            # Rerankerスコア取得
            rerank_result = make_ovms_reranker_request([QUESTION]*len(PASSAGES), PASSAGES)
            rerank_scores = [0.0] * len(PASSAGES)
            for result in rerank_result["results"]:
                rerank_scores[result["index"]] = float(result["relevance_score"])

            rows = [
                ScoreRow(passage=p, embed_score=e, reranker_score=r)
                for p, e, r in zip(PASSAGES, embed_scores, rerank_scores)
            ]

            # Combined sort: prioritize reranker signal, then embedding similarity.
            rows.sort(key=lambda x: (x.reranker_score, x.embed_score), reverse=True)

            print(f"Question: {QUESTION}\n")
            for i, row in enumerate(rows, start=1):
                print(f"[{i}] embed_cos={row.embed_score:.4f}  reranker_sigmoid={row.reranker_score:.4f}")
                print(row.passage)
                print()
    finally:
        if embed_proc:
            embed_proc.terminate()
            print("OVMS (bge) terminated.")
        if reranker_proc:
            reranker_proc.terminate()
            print("OVMS (bge-reranker) terminated.")

if __name__ == "__main__":
    main()
