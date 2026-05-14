from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import openvino as ov
from openvino_tokenizers import convert_tokenizer
from optimum.intel import OVModelForSequenceClassification
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging


MODEL_DIR = Path("models")
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_OUTPUT_DIR = MODEL_DIR / f"{RERANKER_MODEL_ID.split('/')[-1]}-ov"
transformers_logging.set_verbosity_error()


def save_tokenizer_ir(tokenizer, output_dir: Path) -> None:
    tokenizer_ir = convert_tokenizer(tokenizer, number_of_inputs=2, max_length=512)
    ov.save_model(tokenizer_ir, output_dir / "openvino_tokenizer.xml")


def convert_model(output_dir: Path, force: bool = False) -> Path:
    model_xml = output_dir / "openvino_model.xml"
    tokenizer_xml = output_dir / "openvino_tokenizer.xml"
    if model_xml.exists() and tokenizer_xml.exists() and not force:
        print(f"OpenVINO model already exists: {output_dir}")
        return output_dir

    if model_xml.exists() and not tokenizer_xml.exists() and not force:
        print(f"Adding OpenVINO tokenizer IR: {output_dir}")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        save_tokenizer_ir(tokenizer, output_dir)
        print(f"Saved OpenVINO tokenizer: {output_dir}")
        return output_dir

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {RERANKER_MODEL_ID} -> {output_dir} ...")

    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_ID)
    model = OVModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL_ID,
        export=True,
        compile=False,
    )

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    save_tokenizer_ir(tokenizer, output_dir)

    print(f"Saved OpenVINO model: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BAAI/bge-reranker-v2-m3 to OpenVINO IR for local tests."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and recreate the output directory.",
    )
    args = parser.parse_args()

    convert_model(args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
