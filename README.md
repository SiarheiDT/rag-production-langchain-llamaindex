# Production RAG System (LangChain + LlamaIndex)

## Overview
End-to-end implementation of Retrieval-Augmented Generation (RAG) systems.

## Modules
- LC_01_RAG_Foundations
- LC_02_Retrieval_Optimization
- LC_03_RAG_Agent_Systems
- LC_04_RAG_Evaluation_Observability

## Tech Stack
- LangChain
- LlamaIndex
- OpenAI
- DeepLake

## Goal
Build production-ready RAG pipelines with evaluation and agents.

## run.py
1. Setup
python run.py --module 2 --part 1 --task 1
2. Download data
python run.py --module 2 --part 1 --task 2
3. ⚠️ НЕ запускать (utility)
python run.py --module 2 --part 1 --task 3
4. Load documents
python run.py --module 2 --part 1 --task 4
5. Chunking
python run.py --module 2 --part 1 --task 5 -- \
  --chunk-size 512 \
  --chunk-overlap 64
6. Create vector store
python run.py --module 2 --part 1 --task 6 -- \
  --org-id siarhei \
  --dataset-name pg_essay \
  --overwrite
7. Build index
python run.py --module 2 --part 1 --task 7 -- \
  --org-id siarhei \
  --dataset-name pg_essay \
  --data-dir LC_02_Retrieval_Optimization/Part_1/data/paul_graham
8. Basic RAG
python run.py --module 2 --part 1 --task 8 -- \
  --org-id siarhei \
  --dataset-name pg_essay \
  --data-dir LC_02_Retrieval_Optimization/Part_1/data/paul_graham
9. Sub-question engine
python run.py --module 2 --part 1 --task 9 -- \
  --org-id siarhei \
  --dataset-name pg_essay \
  --data-dir LC_02_Retrieval_Optimization/Part_1/data/paul_graham
10. Cohere rerank (basic)
python run.py --module 2 --part 1 --task 10
11. Cohere rerank (LlamaIndex)
python run.py --module 2 --part 1 --task 11 -- \
  --org-id siarhei \
  --dataset-name pg_essay \
  --data-dir LC_02_Retrieval_Optimization/Part_1/data/paul_graham