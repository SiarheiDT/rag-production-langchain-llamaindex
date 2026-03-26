# 🎯 README_INTERVIEW.md  
## RAG Evaluation — How to Answer in Interviews (+ How to Run)

---

## ▶️ How to Run Scripts (IMPORTANT)

```bash
# Retrieval metrics
python run.py --module 4 --part 1 --task 1

# Golden dataset
python run.py --module 4 --part 1 --task 2

# Faithfulness vs relevance
python run.py --module 4 --part 1 --task 3

# LlamaIndex evaluator
python run.py --module 4 --part 1 --task 4

# RAGAS evaluation (clean)
python run.py --module 4 --part 1 --task 5

# Batch evaluation
python run.py --module 4 --part 1 --task 6

# End-to-end pipeline
python run.py --module 4 --part 1 --task 7   --docs-dir LC_04_RAG_Evaluation_Observability/Part_1/data/sample_docs   --dataset LC_04_RAG_Evaluation_Observability/Part_1/data/sample_golden_dataset.json
```

---

## 🧠 Core Strategy (How to Answer)

1. Start high-level (system view)
2. Split into retrieval vs generation
3. Add metrics
4. Mention failure modes
5. Add production practices

---

## 🧩 Canonical Answer Structure

**1. Define RAG**
RAG combines retrieval + generation.

**2. Split problem**
- Retrieval quality
- Generation quality

**3. Metrics**
- Retrieval → Hit Rate, MRR, NDCG
- Generation → Faithfulness, Relevance

**4. Evaluation**
- Golden dataset
- Batch evaluation
- LLM-as-a-judge

**5. Production**
- monitoring
- versioning
- experiment tracking

---

## ❓ Top Questions & Answers

### Q1: What is faithfulness?
Faithfulness means the answer is grounded in retrieved context.

---

### Q2: Faithfulness vs Relevance?
- Faithfulness → correctness vs context  
- Relevance → usefulness vs question  

---

### Q3: Why retrieval matters?
If retrieval fails → hallucinations increase.

---

### Q4: What is MRR?
Measures how early the first relevant result appears.

---

### Q5: Why batch evaluation?
Single query is unreliable → need aggregate metrics.

---

## ⚠️ Common Mistakes

- mixing faithfulness & relevance
- ignoring retrieval metrics
- no golden dataset
- manual-only evaluation

---

## 🧠 Strong Answer Example

"RAG evaluation should be split into retrieval and generation.

For retrieval, I use metrics like Hit Rate and MRR.

For generation, I evaluate faithfulness and relevance.

I use a golden dataset and batch evaluation.

In production, I log queries, context, and scores, and monitor trends."

---

## 🏁 Final Tip

Always:
- structure answer
- use metrics
- give example
- mention production
