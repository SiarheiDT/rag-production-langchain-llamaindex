Module 2 — Advanced RAG Techniques with LlamaIndex
📌 Overview

This module implements a production-grade Retrieval-Augmented Generation (RAG) pipeline using:

LlamaIndex — orchestration layer
DeepLake — vector storage
OpenAI — embeddings + LLM
Cohere — reranking

The goal is to move from basic retrieval → optimized retrieval → intelligent query decomposition.

🧠 Architecture
Raw Data (Paul Graham Essay)
        ↓
Document Loader
        ↓
Text Chunking (Nodes)
        ↓
Embeddings (OpenAI)
        ↓
Vector Store (DeepLake)
        ↓
Vector Index
        ↓
Query Engine (RAG)
        ↓
+ Reranking (Cohere)
+ Sub-question Decomposition
📂 Project Structure
LC_02_Retrieval_Optimization/
└── Part_1/
    ├── m_02_001_requirements_setup.py
    ├── m_02_002_download_paul_graham_data.py
    ├── m_02_003_common.py
    ├── m_02_004_load_documents.py
    ├── m_02_005_create_nodes.py
    ├── m_02_006_create_deeplake_vector_store.py
    ├── m_02_007_build_vector_index.py
    ├── m_02_008_basic_query_engine_streaming.py
    ├── m_02_009_subquestion_query_engine_v2.py
    ├── m_02_010_cohere_rerank_basic.py
    ├── m_02_011_cohere_rerank_llamaindex.py
    └── data/
⚙️ Environment Setup
1. Create virtual environment
python3 -m venv .venv_lc2
source .venv_lc2/bin/activate
pip install --upgrade pip
2. Install dependencies
pip install \
  "llama-index==0.14.18" \
  "llama-index-llms-openai==0.7.3" \
  "llama-index-embeddings-openai==0.6.0" \
  "llama-index-vector-stores-deeplake==0.5.0" \
  "llama-index-postprocessor-cohere-rerank" \
  "deeplake>=4.0.0" \
  "openai>=1.0.0" \
  "tiktoken" \
  "python-dotenv" \
  "cohere"
3. Environment variables (.env)
OPENAI_API_KEY=...
ACTIVELOOP_TOKEN=...
COHERE_API_KEY=...
4. PYTHONPATH (IMPORTANT)
export PYTHONPATH=/home/siarhei/rag-production-langchain-llamaindex
▶️ Execution Flow
1. Setup & Data Download
Install / prepare environment
python m_02_001_requirements_setup.py
Download dataset
python m_02_002_download_paul_graham_data.py
2. Load Documents
python m_02_004_load_documents.py
Output
Loaded document preview
Raw text verification
3. Create Nodes (Chunking)
python m_02_005_create_nodes.py \
  --chunk-size 512 \
  --chunk-overlap 64
Output
Created nodes: 43
Insight
Chunk size impacts retrieval granularity
Overlap prevents context loss
4. Create Vector Store (DeepLake)
python m_02_006_create_deeplake_vector_store.py \
  --org-id siarhei \
  --dataset-name pg_essay \
  --overwrite
Output
Vector store ready: hub://siarhei/pg_essay
5. Build Vector Index
python m_02_007_build_vector_index.py \
  --org-id siarhei \
  --dataset-name pg_essay \
  --data-dir LC_02_Retrieval_Optimization/Part_1/data/paul_graham
Output
Index built successfully.
Uploaded nodes: 43
6. Basic Query Engine (Streaming)
python m_02_008_basic_query_engine_streaming.py \
  --org-id siarhei \
  --dataset-name pg_essay \
  --data-dir LC_02_Retrieval_Optimization/Part_1/data/paul_graham
Output
Paul Graham organizes a summer program...
Insight
Classic RAG pipeline
Retrieval + synthesis
7. Sub-Question Query Engine (Advanced)
python m_02_009_subquestion_query_engine_v2.py \
  --org-id siarhei \
  --dataset-name pg_essay \
  --data-dir LC_02_Retrieval_Optimization/Part_1/data/paul_graham
Output
Generated 3 sub questions.

Q: What did Paul Graham work on before YC
Q: What did Paul Graham work on during YC
Q: What did Paul Graham work on after YC
Insight
Decomposes complex queries
Improves reasoning accuracy
8. Cohere Rerank (Standalone)
python m_02_010_cohere_rerank_basic.py
Output
Rank 1 → relevance: 0.87
Rank 2 → relevance: 0.24
Insight
Filters noisy retrieval
Improves precision
9. Cohere Rerank in LlamaIndex
python m_02_011_cohere_rerank_llamaindex.py \
  --org-id siarhei \
  --dataset-name pg_essay \
  --data-dir LC_02_Retrieval_Optimization/Part_1/data/paul_graham
Output
Sam Altman was involved...
Insight
Rerank integrated into pipeline
Production-ready pattern
🧩 Key Techniques Covered
Technique	Purpose
Chunking	Control retrieval granularity
Vector Store	Persistent semantic search
Query Engine	RAG inference
Sub-questions	Multi-step reasoning
Reranking	Precision improvement
⚠️ Known Issues & Fixes
1. ModuleNotFoundError: common
export PYTHONPATH=project_root
2. .env not loaded

Fix:

load_env()
3. DeepLake S3 warning
INVALID_ACCESS_KEY_ID

👉 Ignore (non-blocking)

4. Dependency conflicts (LlamaIndex)

👉 Avoid installing:

llama-index-question-gen-openai

Use custom version:

m_02_009_subquestion_query_engine_v2.py
🧠 Key Learnings (Senior Level)
1. Retrieval ≠ solved problem
Chunk size directly affects answer quality
Top-k alone is insufficient
2. Reranking is critical
First-pass retrieval is noisy
Rerank significantly improves precision
3. Query decomposition = reasoning
Complex queries → multiple retrieval passes
Improves grounding
4. Data pipeline matters more than LLM
Embeddings quality
Chunking strategy
Index design
5. Production RAG = layered system
Retrieval → Filtering → Reasoning → Generation