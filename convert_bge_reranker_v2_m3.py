from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import openvino as ov
from openvino_tokenizers import convert_tokenizer
from optimum.intel import OVModelForSequenceClassification, OVWeightQuantizationConfig
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging


MODEL_DIR = Path("models")
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_NAME = RERANKER_MODEL_ID.split("/")[-1]
DEFAULT_OUTPUT_DIRS = {
    "fp32": MODEL_DIR / f"{RERANKER_MODEL_NAME}-ov",
    "int8": MODEL_DIR / f"{RERANKER_MODEL_NAME}-int8-ov",
}
transformers_logging.set_verbosity_error()


if os.name == "nt":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def save_tokenizer_ir(tokenizer, output_dir: Path) -> None:
    tokenizer_ir = convert_tokenizer(tokenizer, number_of_inputs=2, max_length=512)
    ov.save_model(tokenizer_ir, output_dir / "openvino_tokenizer.xml")


def convert_model(output_dir: Path, precision: str, force: bool = False) -> Path:
    model_xml = output_dir / "openvino_model.xml"
    tokenizer_xml = output_dir / "openvino_tokenizer.xml"
    if model_xml.exists() and tokenizer_xml.exists() and not force:
        print(f"OpenVINO {precision} model already exists: {output_dir}")
        return output_dir

    if model_xml.exists() and not tokenizer_xml.exists() and not force:
        print(f"Adding OpenVINO tokenizer IR: {output_dir}")
        tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_ID)
        tokenizer.save_pretrained(output_dir)
        save_tokenizer_ir(tokenizer, output_dir)
        print(f"Saved OpenVINO tokenizer: {output_dir}")
        return output_dir

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {RERANKER_MODEL_ID} -> {output_dir} ({precision}) ...")

    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_ID)
    quantization_config = None
    if precision == "int8":
        quantization_config = OVWeightQuantizationConfig(bits=8)

    model = OVModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL_ID,
        export=True,
        compile=False,
        quantization_config=quantization_config,
    )

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    save_tokenizer_ir(tokenizer, output_dir)

    print(f"Saved OpenVINO {precision} model: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BAAI/bge-reranker-v2-m3 to OpenVINO IR for local tests."
    )
    parser.add_argument(
        "--precision",
        choices=["all", "fp32", "int8"],
        default="all",
        help="Which precision to convert. Default: all",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MODEL_DIR,
        help=f"Output root directory. Default: {MODEL_DIR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and recreate the output directory.",
    )
    args = parser.parse_args()

    precisions = ["fp32", "int8"] if args.precision == "all" else [args.precision]
    for precision in precisions:
        output_dir = args.output_root / DEFAULT_OUTPUT_DIRS[precision].name
        convert_model(output_dir, precision=precision, force=args.force)


if __name__ == "__main__":
    main()
