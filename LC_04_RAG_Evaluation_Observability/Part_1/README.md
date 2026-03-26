# Module 4 — RAG Evaluation & Observability Practice Scripts

This folder contains a compact Python practice pack for reinforcing the core ideas from **Module 4: Retrieval-Augmented Generation Evaluation and Observability**.

The goal is not only to read about RAG evaluation, but to convert the theory into repeatable scripts that make the following concepts concrete:

- retrieval metrics
- golden datasets
- faithfulness vs relevance
- batch evaluation
- LlamaIndex evaluators
- RAGAS-based evaluation patterns

---

## Files

### `01_retrieval_metrics_demo.py`
Implements core retrieval metrics from scratch:

- Hit Rate@k
- MRR
- AP / MAP
- NDCG

Use this file to understand the retrieval side of RAG independently from the LLM.

---

### `02_golden_dataset_template.py`
Creates a starter `golden_dataset.jsonl`.

Use this file to learn how to structure evaluation benchmarks around:

- question
- expected source IDs
- reference answer
- notes

This is the foundation for stable regression testing in RAG systems.

---

### `03_faithfulness_vs_relevance_demo.py`
A concept-first script that shows the difference between:

- **faithfulness** = grounded in context
- **relevance** = answers the question

This distinction is one of the most common interview topics in RAG evaluation.

---

### `04_llamaindex_faithfulness_eval.py`
A minimal LlamaIndex example using `FaithfulnessEvaluator`.

Use this script when you want to move from theory into evaluator-driven inspection of model outputs.

---

### `05_ragas_eval_pipeline.py`
A minimal RAGAS scaffold.

Demonstrates how to prepare an evaluation dataset and run metrics such as:

- faithfulness
- answer relevancy
- context precision
- context recall

---

### `06_batch_eval_runner_template.py`
Shows the production mindset:

- evaluate multiple queries
- aggregate pass rates
- compare evaluation results over time

This is closer to how RAG assessment should work in real systems.

---

## Suggested Learning Order

1. Start with `03_faithfulness_vs_relevance_demo.py`
2. Then run `01_retrieval_metrics_demo.py`
3. Create your own benchmark via `02_golden_dataset_template.py`
4. Move to `04_llamaindex_faithfulness_eval.py`
5. Then study `05_ragas_eval_pipeline.py`
6. Finish with `06_batch_eval_runner_template.py`

---

## What to Memorize for Interviews

### Retrieval layer
You should be able to explain:

- why retrieval failure breaks the whole RAG system
- what Hit Rate measures
- what MRR measures
- when MAP and NDCG are useful

### Generation layer
You should be able to explain:

- what faithfulness means
- how faithfulness differs from answer relevance
- why hallucinations are dangerous in RAG
- why a faithful answer can still be a bad answer

### Evaluation design
You should be able to explain:

- why single-query testing is weak
- why batch evaluation is necessary
- why golden datasets should come from real user questions
- why synthetic questions are useful for prototyping but weaker for user-centric evaluation

---

## Practical Advice

In real projects:

- use a stronger model for evaluation than for generation when budget allows
- version your golden dataset
- log evaluation results by experiment name
- compare runs after changing chunking, retriever config, prompts, or embedding model
- separate retrieval evaluation from end-to-end answer evaluation

---

## Minimal Setup Notes

Some scripts are fully offline and require no API keys.
Some scripts require OpenAI and ecosystem libraries such as:

- `llama-index`
- `ragas`
- `datasets`

If a script needs an API key, it checks for `OPENAI_API_KEY` and exits clearly when missing.

---

## Recommended Next Step

After you review these scripts, the best continuation is to build one more file:

- a **single end-to-end evaluation harness** that:
  - loads documents
  - chunks them
  - runs retrieval
  - generates answers
  - evaluates retrieval metrics
  - evaluates faithfulness/relevance
  - writes results to CSV or JSON

That would be the first interview-ready, portfolio-grade artifact for this module.
