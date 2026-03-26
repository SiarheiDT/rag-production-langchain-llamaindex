"""
02_golden_dataset_template.py

Creates a starter Golden Dataset for RAG evaluation in JSONL format.

What this script does:
- Builds a small example golden dataset.
- Saves it to ./golden_dataset.jsonl
- Shows how to structure evaluation data for:
  * question
  * expected source IDs
  * optional reference answer

Why it matters:
A Golden Dataset is the benchmark that lets you compare retrieval and
answer quality over time. In production, this dataset should be built from
real user questions, domain expert validation, and version-controlled curation.

Run:
    python 02_golden_dataset_template.py
"""

from __future__ import annotations

import json
from pathlib import Path


def build_records() -> list[dict]:
    return [
        {
            "question": "What is faithfulness in a RAG system?",
            "expected_source_ids": ["rag_metrics_01"],
            "reference_answer": "Faithfulness measures whether the generated answer is grounded in the retrieved context and does not introduce fabrication.",
            "notes": "Good baseline question for hallucination control.",
        },
        {
            "question": "Why is MRR useful in retrieval evaluation?",
            "expected_source_ids": ["retrieval_metrics_02"],
            "reference_answer": "MRR measures how early the first relevant result appears in ranked retrieval outputs.",
            "notes": "Useful when only the first strong result matters.",
        },
        {
            "question": "What is the difference between faithfulness and answer relevance?",
            "expected_source_ids": ["rag_metrics_01", "eval_design_03"],
            "reference_answer": "Faithfulness checks grounding in retrieved context, while answer relevance checks whether the answer actually addresses the user question.",
            "notes": "Common interview question.",
        },
    ]


def save_jsonl(records: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    output_path = Path("golden_dataset.jsonl")
    records = build_records()
    save_jsonl(records, output_path)

    print(f"Saved {len(records)} records to: {output_path.resolve()}")
    print("\nExample record:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))
    print("\nNext step:")
    print("Replace toy questions with real user queries and expert-validated sources.")


if __name__ == "__main__":
    main()
