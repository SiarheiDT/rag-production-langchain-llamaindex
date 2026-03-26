"""
07_end_to_end_rag_eval_pipeline.py

Production-style end-to-end Retrieval-Augmented Generation (RAG) evaluation pipeline.

What this script demonstrates
-----------------------------
1. Load local text documents from a directory.
2. Split them into chunks.
3. Build a simple vector index with embeddings.
4. Generate answers with retrieved context.
5. Evaluate retrieval quality with ranking metrics.
6. Evaluate answer quality with heuristic metrics.
7. Optionally run LLM-based evaluators via RAGAS / LlamaIndex.
8. Save a structured JSON report for later analysis.

Why this script matters
-----------------------
A strong RAG system is not judged only by whether it "answers".
It should be evaluated across multiple layers:
- retrieval correctness
- ranking quality
- context coverage
- faithfulness / grounding
- answer relevance
- operational observability

This file is designed as a learning and portfolio-ready template.
It is intentionally verbose and heavily commented.

Recommended usage
-----------------
python run.py --module 4 --part 1 --task 7 -- \
  --docs-dir LC_04_RAG_Evaluation_Observability/Part_1/data/sample_docs \
  --dataset LC_04_RAG_Evaluation_Observability/Part_1/data/sample_golden_dataset.json

Optional LLM-based evaluation
-----------------------------
Set environment variables before running:
- OPENAI_API_KEY=...

Then add:
--enable-ragas
--enable-llamaindex-eval

Notes
-----
- The default execution path works without paid APIs.
- Local metrics are heuristic but useful for understanding the pipeline.
- LLM-based evaluation is optional because it requires external services.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common.env_loader import load_env

# -----------------------------------------------------------------------------
# Optional dependencies for embeddings / vector search
# -----------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires scikit-learn. Install it with: pip install scikit-learn"
    ) from exc


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
@dataclass
class Document:
    """Represents a source document loaded from disk."""

    doc_id: str
    path: str
    text: str


@dataclass
class Chunk:
    """Represents a chunk derived from a source document."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int


@dataclass
class GoldenExample:
    """Represents one evaluation example from the golden dataset."""

    question: str
    expected_answer: Optional[str] = None
    expected_doc_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Represents a retrieved chunk with similarity score."""

    chunk_id: str
    doc_id: str
    score: float
    text: str


@dataclass
class QueryEvaluation:
    """Per-query evaluation record."""

    question: str
    expected_answer: Optional[str]
    expected_doc_ids: List[str]
    predicted_answer: str
    retrieved_doc_ids: List[str]
    retrieved_chunk_ids: List[str]
    top_k_scores: List[float]
    retrieval_metrics: Dict[str, float]
    answer_metrics: Dict[str, float]
    notes: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize text for lightweight lexical comparison."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Simple tokenizer used for transparent metric computation."""
    return WORD_RE.findall(normalize_text(text))


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[Tuple[int, int, str]]:
    """
    Chunk text using character windows.

    This is intentionally simple and explainable.
    In a real production system you may switch to sentence-aware or token-aware
    chunking, but the evaluation concepts stay the same.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Tuple[int, int, str]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        if end == text_len:
            break
        start = end - overlap

    return chunks


# -----------------------------------------------------------------------------
# File loading
# -----------------------------------------------------------------------------
def load_documents(docs_dir: Path) -> List[Document]:
    """Load all .txt and .md files from the provided directory recursively."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_dir}")

    docs: List[Document] = []
    supported_suffixes = {".txt", ".md"}

    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported_suffixes:
            text = path.read_text(encoding="utf-8")
            docs.append(
                Document(
                    doc_id=path.stem,
                    path=str(path),
                    text=text,
                )
            )

    if not docs:
        raise ValueError(f"No .txt or .md files found under: {docs_dir}")

    return docs


def build_chunks(documents: Sequence[Document], chunk_size: int, overlap: int) -> List[Chunk]:
    """Split all documents into chunk records."""
    chunks: List[Chunk] = []

    for doc in documents:
        for idx, (start, end, chunk_text_value) in enumerate(
            chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap),
            start=1,
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}__chunk_{idx}",
                    doc_id=doc.doc_id,
                    text=chunk_text_value,
                    start_char=start,
                    end_char=end,
                )
            )

    return chunks


# -----------------------------------------------------------------------------
# Golden dataset loading
# -----------------------------------------------------------------------------
def load_golden_dataset(dataset_path: Path) -> List[GoldenExample]:
    """
    Load a golden dataset from JSON.

    Expected format:
    [
      {
        "question": "...",
        "expected_answer": "...",
        "expected_doc_ids": ["doc1"]
      }
    ]
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Golden dataset JSON must be a list of objects")

    examples: List[GoldenExample] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item #{idx} must be an object")
        if "question" not in item:
            raise ValueError(f"Dataset item #{idx} is missing 'question'")

        examples.append(
            GoldenExample(
                question=item["question"],
                expected_answer=item.get("expected_answer"),
                expected_doc_ids=list(item.get("expected_doc_ids", [])),
                metadata=dict(item.get("metadata", {})),
            )
        )

    if not examples:
        raise ValueError("Golden dataset is empty")

    return examples


# -----------------------------------------------------------------------------
# Vector index
# -----------------------------------------------------------------------------
class SimpleVectorIndex:
    """
    Minimal vector index built on top of TF-IDF.

    This is not intended to replace production embedding systems.
    It exists to make the evaluation workflow transparent and reproducible.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("Cannot build vector index with zero chunks")

        self.vectorizer = TfidfVectorizer()
        self.chunk_texts = [chunk.text for chunk in self.chunks]
        self.chunk_matrix = self.vectorizer.fit_transform(self.chunk_texts)

    def search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Retrieve top-k chunks by cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_matrix)[0]
        ranked_indices = similarities.argsort()[::-1][:top_k]

        results: List[RetrievalResult] = []
        for idx in ranked_indices:
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=float(similarities[int(idx)]),
                    text=chunk.text,
                )
            )
        return results


# -----------------------------------------------------------------------------
# Simple answer generation
# -----------------------------------------------------------------------------
def generate_answer_from_context(question: str, retrieved: Sequence[RetrievalResult], max_context_chars: int = 1200) -> str:
    """
    Build a deterministic answer from retrieved context.

    This is a deliberately lightweight stand-in for an LLM.
    It concatenates the strongest retrieved snippets so the evaluation pipeline can
    still be executed end-to-end without external APIs.
    """
    if not retrieved:
        return "No answer could be generated because no relevant context was retrieved."

    context_parts: List[str] = []
    total_chars = 0

    for item in retrieved:
        snippet = item.text.strip().replace("\n", " ")
        snippet = re.sub(r"\s+", " ", snippet)
        remaining = max_context_chars - total_chars
        if remaining <= 0:
            break
        snippet = snippet[:remaining]
        context_parts.append(snippet)
        total_chars += len(snippet)

    context_text = " ".join(context_parts).strip()
    return f"Question: {question}\nAnswer based on retrieved context: {context_text}"


# -----------------------------------------------------------------------------
# Retrieval metrics
# -----------------------------------------------------------------------------
def reciprocal_rank(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]) -> float:
    """Return reciprocal rank for the first relevant result."""
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int) -> float:
    """Return 1.0 if at least one relevant document is found in top-k, else 0.0."""
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0
    return 1.0 if any(doc_id in relevant_set for doc_id in retrieved_doc_ids[:k]) else 0.0


def average_precision(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]) -> float:
    """Compute AP for ranked retrieval results."""
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return 0.0
    return precision_sum / len(relevant_set)


def ndcg_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int) -> float:
    """Compute NDCG@k with binary relevance."""
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0

    dcg = 0.0
    for idx, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_set else 0.0
        if rel > 0:
            dcg += rel / math.log2(idx + 1)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# -----------------------------------------------------------------------------
# Answer quality metrics
# -----------------------------------------------------------------------------
def lexical_overlap_ratio(reference: str, candidate: str) -> float:
    """Compute token overlap ratio relative to the reference token set."""
    ref_tokens = set(tokenize(reference))
    cand_tokens = set(tokenize(candidate))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between token sets."""
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def estimate_faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """
    Heuristic faithfulness estimate.

    Measures how much of the answer vocabulary appears in the retrieved context.
    This is a transparent proxy, not a replacement for LLM-based grounding checks.
    """
    answer_tokens = set(tokenize(answer))
    context_tokens = set(tokenize(" ".join(contexts)))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def estimate_answer_relevance(question: str, answer: str) -> float:
    """Heuristic relevance estimate via lexical overlap between question and answer."""
    return lexical_overlap_ratio(question, answer)


def estimate_context_precision(question: str, contexts: Sequence[str]) -> float:
    """
    Heuristic context precision.

    Measures how much of the retrieved context lexically aligns with the question.
    """
    if not contexts:
        return 0.0
    scores = [jaccard_similarity(question, ctx) for ctx in contexts]
    return float(sum(scores) / len(scores))


def estimate_context_recall(expected_answer: Optional[str], contexts: Sequence[str]) -> float:
    """
    Heuristic context recall.

    If a reference answer exists, estimate how much of that answer is represented in
    the retrieved context. If there is no reference answer, return 0.0.
    """
    if not expected_answer:
        return 0.0
    return lexical_overlap_ratio(expected_answer, " ".join(contexts))


# -----------------------------------------------------------------------------
# Optional LLM-based evaluators
# -----------------------------------------------------------------------------
def try_ragas_evaluation(
    question: str,
    answer: str,
    contexts: Sequence[str],
    ground_truth: Optional[str],
) -> Dict[str, Any]:
    """
    Optionally run a very small RAGAS evaluation for one sample.

    This function is defensive:
    - it returns an explanatory payload if dependencies are missing
    - it returns an explanatory payload if API credentials are unavailable
    """
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "skipped", "reason": "OPENAI_API_KEY not set"}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except Exception as exc:
        return {"status": "skipped", "reason": f"RAGAS dependencies unavailable: {exc}"}

    try:
        payload = {
            "question": [question],
            "answer": [answer],
            "contexts": [list(contexts)],
            "ground_truth": [ground_truth or ""],
        }
        dataset = Dataset.from_dict(payload)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return {"status": "ok", "scores": result.to_pandas().to_dict(orient="records")[0]}
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "reason": str(exc)}


def try_llamaindex_evaluation(question: str, answer: str, contexts: Sequence[str]) -> Dict[str, Any]:
    """
    Optionally run LlamaIndex evaluators.

    This path is informative, but optional.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "skipped", "reason": "OPENAI_API_KEY not set"}

    try:
        from llama_index.core import Document as LIDocument
        from llama_index.core import ServiceContext, VectorStoreIndex
        from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
        from llama_index.llms.openai import OpenAI
    except Exception as exc:
        return {"status": "skipped", "reason": f"LlamaIndex dependencies unavailable: {exc}"}

    try:
        # A tiny temporary index created only for evaluation context.
        docs = [LIDocument(text=ctx) for ctx in contexts]
        llm = OpenAI(model="gpt-4o-mini", temperature=0)
        service_context = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(question)

        faithfulness_eval = FaithfulnessEvaluator(service_context=service_context)
        relevancy_eval = RelevancyEvaluator(service_context=service_context)

        faithfulness_result = faithfulness_eval.evaluate_response(response=response)
        relevancy_result = relevancy_eval.evaluate_response(query=question, response=response)

        return {
            "status": "ok",
            "scores": {
                "faithfulness_passing": bool(faithfulness_result.passing),
                "relevancy_passing": bool(relevancy_result.passing),
                "generated_eval_answer": str(response),
            },
        }
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "reason": str(exc)}


# -----------------------------------------------------------------------------
# Main evaluation routine
# -----------------------------------------------------------------------------
def evaluate_rag_system(
    docs_dir: Path,
    dataset_path: Path,
    output_path: Path,
    chunk_size: int,
    overlap: int,
    top_k: int,
    enable_ragas: bool,
    enable_llamaindex_eval: bool,
) -> Dict[str, Any]:
    """Execute the full evaluation pipeline and return a report dictionary."""
    documents = load_documents(docs_dir)
    chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    golden_examples = load_golden_dataset(dataset_path)
    index = SimpleVectorIndex(chunks)

    per_query_results: List[QueryEvaluation] = []

    for example in golden_examples:
        retrieved = index.search(example.question, top_k=top_k)
        answer = generate_answer_from_context(example.question, retrieved)

        retrieved_doc_ids = [item.doc_id for item in retrieved]
        retrieved_chunk_ids = [item.chunk_id for item in retrieved]
        retrieved_contexts = [item.text for item in retrieved]
        top_k_scores = [item.score for item in retrieved]

        retrieval_metrics = {
            "reciprocal_rank": reciprocal_rank(retrieved_doc_ids, example.expected_doc_ids),
            f"hit_rate@{top_k}": hit_rate_at_k(retrieved_doc_ids, example.expected_doc_ids, top_k),
            "average_precision": average_precision(retrieved_doc_ids, example.expected_doc_ids),
            f"ndcg@{top_k}": ndcg_at_k(retrieved_doc_ids, example.expected_doc_ids, top_k),
        }

        answer_metrics = {
            "faithfulness_estimate": estimate_faithfulness(answer, retrieved_contexts),
            "answer_relevance_estimate": estimate_answer_relevance(example.question, answer),
            "context_precision_estimate": estimate_context_precision(example.question, retrieved_contexts),
            "context_recall_estimate": estimate_context_recall(example.expected_answer, retrieved_contexts),
            "reference_overlap_estimate": lexical_overlap_ratio(example.expected_answer or "", answer)
            if example.expected_answer
            else 0.0,
        }

        notes: Dict[str, Any] = {}
        if enable_ragas:
            notes["ragas"] = try_ragas_evaluation(
                question=example.question,
                answer=answer,
                contexts=retrieved_contexts,
                ground_truth=example.expected_answer,
            )
        if enable_llamaindex_eval:
            notes["llamaindex_eval"] = try_llamaindex_evaluation(
                question=example.question,
                answer=answer,
                contexts=retrieved_contexts,
            )

        per_query_results.append(
            QueryEvaluation(
                question=example.question,
                expected_answer=example.expected_answer,
                expected_doc_ids=example.expected_doc_ids,
                predicted_answer=answer,
                retrieved_doc_ids=retrieved_doc_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                top_k_scores=top_k_scores,
                retrieval_metrics=retrieval_metrics,
                answer_metrics=answer_metrics,
                notes=notes,
            )
        )

    summary = aggregate_results(per_query_results)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "docs_dir": str(docs_dir),
            "dataset_path": str(dataset_path),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "enable_ragas": enable_ragas,
            "enable_llamaindex_eval": enable_llamaindex_eval,
        },
        "corpus": {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "document_ids": [doc.doc_id for doc in documents],
        },
        "summary": summary,
        "queries": [asdict(item) for item in per_query_results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


# -----------------------------------------------------------------------------
# Aggregation and reporting
# -----------------------------------------------------------------------------
def safe_mean(values: Iterable[float]) -> float:
    """Mean with empty-sequence safety."""
    values = list(values)
    return float(statistics.mean(values)) if values else 0.0


def aggregate_results(results: Sequence[QueryEvaluation]) -> Dict[str, Any]:
    """Aggregate per-query results into summary metrics."""
    retrieval_metric_names = set()
    answer_metric_names = set()

    for item in results:
        retrieval_metric_names.update(item.retrieval_metrics.keys())
        answer_metric_names.update(item.answer_metrics.keys())

    retrieval_summary = {
        name: safe_mean(item.retrieval_metrics.get(name, 0.0) for item in results)
        for name in sorted(retrieval_metric_names)
    }
    answer_summary = {
        name: safe_mean(item.answer_metrics.get(name, 0.0) for item in results)
        for name in sorted(answer_metric_names)
    }

    failed_hits = []
    for item in results:
        hit_metric_name = next((k for k in item.retrieval_metrics if k.startswith("hit_rate@")), None)
        if hit_metric_name and item.retrieval_metrics.get(hit_metric_name, 0.0) == 0.0:
            failed_hits.append(item.question)

    return {
        "query_count": len(results),
        "retrieval_metrics": retrieval_summary,
        "answer_metrics": answer_summary,
        "questions_with_no_hit": failed_hits,
    }


def print_console_summary(report: Dict[str, Any]) -> None:
    """Print a short but useful terminal summary."""
    summary = report["summary"]
    retrieval_metrics = summary["retrieval_metrics"]
    answer_metrics = summary["answer_metrics"]

    print("=" * 80)
    print("RAG Evaluation Summary")
    print("=" * 80)
    print(f"Created at (UTC): {report['created_at_utc']}")
    print(f"Documents: {report['corpus']['document_count']}")
    print(f"Chunks:    {report['corpus']['chunk_count']}")
    print(f"Queries:   {summary['query_count']}")
    print("-" * 80)
    print("Retrieval metrics")
    for name, value in retrieval_metrics.items():
        print(f"  {name:24s}: {value:.4f}")
    print("-" * 80)
    print("Answer metrics")
    for name, value in answer_metrics.items():
        print(f"  {name:24s}: {value:.4f}")
    print("-" * 80)
    if summary["questions_with_no_hit"]:
        print("Questions with no relevant document hit:")
        for question in summary["questions_with_no_hit"]:
            print(f"  - {question}")
    else:
        print("All questions had at least one relevant document in top-k.")
    print("=" * 80)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end RAG evaluation pipeline with retrieval and answer metrics."
    )
    parser.add_argument("--docs-dir", type=Path, required=True, help="Directory with .txt/.md source documents")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to golden dataset JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/rag_eval_report.json"),
        help="Path to output JSON report",
    )
    parser.add_argument("--chunk-size", type=int, default=700, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=120, help="Chunk overlap in characters")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k retrieval depth")
    parser.add_argument(
        "--enable-ragas",
        action="store_true",
        help="Enable optional RAGAS evaluation if dependencies and API key are available",
    )
    parser.add_argument(
        "--enable-llamaindex-eval",
        action="store_true",
        help="Enable optional LlamaIndex evaluators if dependencies and API key are available",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    load_env()
    
    try:
        report = evaluate_rag_system(
            docs_dir=args.docs_dir,
            dataset_path=args.dataset,
            output_path=args.output,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            top_k=args.top_k,
            enable_ragas=args.enable_ragas,
            enable_llamaindex_eval=args.enable_llamaindex_eval,
        )
        print_console_summary(report)
        print(f"Report saved to: {args.output}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
