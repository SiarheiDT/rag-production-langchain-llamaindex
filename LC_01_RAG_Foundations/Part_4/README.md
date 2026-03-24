# Part 4 — Chat with Your Code (LlamaIndex + Deep Lake + GitHub)

These scripts were generated from the lesson:
`2.4.Chat with Your Code_LlamaIndex and Activeloop Deep Lake for GitHub Repositories.txt`

## Included scripts

1. `m_04_001_github_quickstart.py`
   - High-level interactive chat with a GitHub repository.

2. `m_04_002_github_index_once.py`
   - Index a repository and ask a single question.

3. `m_04_003_retriever_topk_demo.py`
   - Inspect retrieved nodes with `similarity_top_k`.

4. `m_04_004_custom_query_engine.py`
   - Custom low-level query engine using retriever, response synthesizer, and similarity cutoff.

5. `m_04_005_response_modes_demo.py`
   - Compare `default`, `compact`, `tree_summarize`, and `no_text` response modes.

6. `m_04_006_local_vs_cloud_dataset_path.py`
   - Helper script to validate local vs cloud Deep Lake dataset paths.

## Requirements

Recommended packages:
- llama-index
- llama-index-readers-github
- llama-index-vector-stores-deeplake
- deeplake
- openai
- python-dotenv
- nest_asyncio

## Environment variables

Copy `.env.example` to `.env` and fill in:
- `OPENAI_API_KEY`
- `ACTIVELOOP_TOKEN`
- `GITHUB_TOKEN`

## Notes

- For faster experimentation, use a local dataset path like `./repository_db`.
- For cloud storage, use a `hub://YOUR_ORG/...` path and a valid `ACTIVELOOP_TOKEN`.
- The GitHub loader works best with a classic personal GitHub token.
