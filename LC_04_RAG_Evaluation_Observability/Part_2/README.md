# LC_04 — RAG Evaluation & Observability  
## Part 2 — LangSmith Tracing + Production-like RAG Pipeline

---

## 📌 Overview

This module demonstrates how to **observe, debug, and improve RAG systems** using:

- LangSmith tracing
- Prompt versioning
- Run metadata
- End-to-end traced RAG pipeline
- Retrieval optimization (MMR + reranking)

---

## 🧠 What You Will Learn

- How to trace LLM applications in production
- How to debug retrieval vs generation issues
- How to compare prompts safely before deployment
- How to improve RAG quality using reranking
- How to structure observable pipelines

---

## 📁 Project Structure

Part_2/
├── data/
│   ├── sample_docs/
│   └── advanced_docs/          # (optional advanced dataset)
│
├── m_02_001_langsmith_env_check.py
├── m_02_002_basic_langsmith_trace.py
├── m_02_003_langsmith_prompt_versioning_demo.py
├── m_02_004_langsmith_traced_rag_pipeline.py
├── m_02_005_langsmith_run_metadata_demo.py
├── m_02_006_langserve_export_stub.py

---

## ⚙️ Setup

### 1. Install dependencies

pip install   langchain   langchain-openai   langchain-community   langchain-text-splitters   faiss-cpu   sentence-transformers

---

### 2. Configure environment

.env:

OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING_V2=true
LANGSMITH_PROJECT=rag-eval
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com

---

## ▶️ How to Run

### 1. Check environment

python run.py --module 4 --part 2 --task 1

---

### 2. Basic trace

python run.py --module 4 --part 2 --task 2

---

### 3. Prompt versioning

python run.py --module 4 --part 2 --task 3

---

### 4. Traced RAG pipeline

python run.py --module 4 --part 2 --task 4 --   --docs-dir LC_04_RAG_Evaluation_Observability/Part_2/data/sample_docs

---

### 5. Metadata demo

python run.py --module 4 --part 2 --task 5

---

## 🔍 What Happens in the RAG Pipeline

documents
→ chunking
→ embeddings
→ FAISS
→ MMR retrieval
→ reranking (cross-encoder)
→ filtered context
→ LLM
→ traced output

---

## 🧪 Key Features

### ✔ Retrieval Optimization

- MMR (diversity-aware retrieval)
- Cross-encoder reranking
- Context filtering (top-k)

---

### ✔ Observability

- Full LangSmith trace
- Retrieval + prompt + output visibility
- Debug prints (before/after rerank)

---

### ✔ Prompt Control

- Strict grounding
- Noise filtering
- Reduced hallucination

---

## 📊 Typical Issues Demonstrated

| Problem            | Solution                    |
|-------------------|----------------------------|
| Mixed topics      | reranking                  |
| Noisy retrieval   | MMR + filtering            |
| Hallucinations    | strict prompt              |
| Debug difficulty  | LangSmith tracing          |

---

## 🚀 Advanced Usage

You can use:

data/advanced_docs/

To test:
- adversarial prompts
- conflicting documents
- noisy retrieval
- paraphrased content

---

## 🧠 Key Insight

> Retrieval quality often matters more than model quality.

---

## ✅ Module Status

| Task | Description                  | Status |
|------|------------------------------|--------|
| 4.2.1 | Env check                   | ✅ |
| 4.2.2 | Basic tracing              | ✅ |
| 4.2.3 | Prompt versioning          | ✅ |
| 4.2.4 | Traced RAG pipeline        | ✅ |
| 4.2.5 | Metadata logging           | ✅ |
| 4.2.6 | LangServe stub             | ⚠️ optional |

---

## 📌 Conclusion

This module upgrades your RAG system from:

toy demo → observable system

and prepares it for:

production debugging and optimization
