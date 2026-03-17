<p align="center">
  <img src="assets/banner.svg" alt="RAGnaros Banner" width="100%"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"/></a>
  <a href="https://pypi.org/project/ragnaros/"><img src="https://img.shields.io/badge/pypi-v0.1.0-orange.svg" alt="PyPI"/></a>
</p>

<p align="center">
  <strong>Adaptive k-selection for RAG pipelines using statistical hypothesis testing</strong>
</p>

<p align="center">
  RAGnaros replaces the fixed <code>k</code> document retrieval parameter in RAG pipelines with a statistically-grounded dynamic selection.<br/>
  Instead of blindly fetching a constant number of documents per query, it estimates how many documents are <em>actually needed</em> — reducing token usage and cost while maintaining or improving answer quality.
</p>

---

## Why RAGnaros?

Standard RAG systems retrieve a fixed number of documents `k` for every query, regardless of how complex or simple the question is. This creates two failure modes:

- **Over-retrieval** — simple queries fetch 10 documents when 1 would suffice, wasting tokens and increasing cost.
- **Under-retrieval** — complex multi-hop questions need more context than a conservative fixed `k` provides.

RAGnaros solves this by treating each query as a statistical hypothesis test: *are there significantly more relevant documents than background noise in this corpus?*

### Research results (HotPotQA, 100 questions)

| Method | Accuracy | Cost (100 questions) | Acc / Dollar |
|---|---|---|---|
| Fixed k=1 | 59% | $0.009 | 65.6 |
| Fixed k=7 (best fixed) | 67% | $0.076 | 8.8 |
| Fixed k=20 | 68% | $0.199 | 3.4 |
| Bonferroni (α=0.05) | 72% | $0.021 | 34.3 |
| Benjamini-Hochberg (α=0.05) | 73% | $0.021 | **34.8** |
| **Higher Criticism (α=0.05)** | **79%** | **$0.040** | **19.8** |

Higher Criticism achieves **+12 percentage points** accuracy over the best fixed-k baseline at **5× lower cost**.

---

## Installation

```bash
# Core library (no visualization or evaluation extras)
pip install ragnaros

# With visualization
pip install "ragnaros[viz]"

# With evaluation utilities (pandas, tqdm)
pip install "ragnaros[eval]"

# Everything including LangChain integrations
pip install "ragnaros[all]"
```

Requires Python 3.11+.

---

## Quick Start

### Step 1 — Build the null distribution (once per corpus)

The null distribution captures how similar a *random, unrelated* query tends to be to your corpus documents. It's built once and cached to disk.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from ragnaros import NullDistribution

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Uses the built-in conversational corpus (no extra data needed)
null_dist = NullDistribution.from_builtin_corpus(
    vectorstore=vectorstore,
    embeddings=embeddings,
    cache_path="./null_dist.npy",  # saved and reused on subsequent runs
)
```

### Step 2 — Create a DynamicRetriever

```python
from ragnaros import DynamicRetriever

retriever = DynamicRetriever.from_vectorstore(
    vectorstore=vectorstore,
    embeddings=embeddings,
    null_distribution=null_dist,
    estimator="higher_criticism",  # recommended
    alpha=0.05,
    max_k=10,
)
```

### Step 3 — Drop into any LangChain chain

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

result = chain.invoke({"query": "Who wrote Les Misérables?"})
```

That's it. The retriever automatically selects the optimal `k` for each query.

---

## Estimators

All three estimators share the same interface and can be swapped via the `estimator` parameter.

### Higher Criticism (recommended)

Based on Donoho & Jin (2004). Finds the index where the gap between observed and expected p-values is maximised. Best for sparse signals where only a few documents are truly relevant.

```python
retriever = DynamicRetriever.from_vectorstore(..., estimator="higher_criticism")
```

### Benjamini-Hochberg

Controls the False Discovery Rate (FDR) at level `alpha`. Balances between precision and recall. Good general-purpose choice.

```python
retriever = DynamicRetriever.from_vectorstore(..., estimator="benjamini_hochberg")
```

### Bonferroni

Controls the Family-Wise Error Rate (FWER). The most conservative estimator — minimises false positives at the cost of potentially under-retrieving.

```python
retriever = DynamicRetriever.from_vectorstore(..., estimator="bonferroni")
```

### Custom estimator

Implement any callable with the signature:

```python
def my_estimator(
    question_emb: np.ndarray,    # shape (D,)
    doc_embs: list[np.ndarray],  # candidates sorted by similarity
    null_distribution: np.ndarray,
    alpha: float,
    max_k: int,
) -> int:
    ...

retriever = DynamicRetriever.from_vectorstore(..., estimator=my_estimator)
```

---

## Retrieval Modes

### Candidates mode (default)

Works with any LangChain vector store. No pre-computation required.

1. Fetches `max_candidates` documents from the vector store.
2. Re-embeds them to compute cosine similarities.
3. Applies the estimator.
4. Returns the top `k` candidates.

```python
retriever = DynamicRetriever.from_vectorstore(
    ...,
    mode="candidates",      # default
    max_candidates=50,
)
```

### Exact mode

Matches the original research methodology. Requires all corpus embeddings pre-loaded in memory — useful for smaller corpora or when you need exact reproducibility.

```python
import numpy as np

# Extract all embeddings from Chroma (Chroma-specific helper)
stored = vectorstore._collection.get(include=["embeddings"])
corpus_embeddings = np.asarray(stored["embeddings"], dtype=np.float32)

retriever = DynamicRetriever.from_vectorstore(
    ...,
    mode="exact",
    corpus_embeddings=corpus_embeddings,
)
```

---

## Null Distribution

The null distribution is the foundation of the statistical testing. It represents "what cosine similarity looks like when a query is unrelated to this corpus."

### Using the built-in corpus (easiest)

```python
null_dist = NullDistribution.from_builtin_corpus(
    vectorstore=vectorstore,
    embeddings=embeddings,
    n_trials=200,       # how many random unrelated texts to sample
    k=20,               # top-k docs compared per trial
    cache_path="./null_dist.npy",
)
```

### Using your own unrelated corpus

```python
my_unrelated_texts = ["random sentence 1", "random sentence 2", ...]
null_dist = NullDistribution.from_corpus(
    texts=my_unrelated_texts,
    vectorstore=vectorstore,
    embeddings=embeddings,
    cache_path="./null_dist.npy",
)
```

### Loading a pre-built distribution

```python
null_dist = NullDistribution.load("./null_dist.npy")
```

### Providing a raw array

```python
import numpy as np
null_dist = NullDistribution.from_array(my_scores_array)
```

---

## Async Support

All retrieval operations are fully async:

```python
docs = await retriever.ainvoke(query)

# In an LCEL chain
chain = retriever | format_docs | prompt | llm | StrOutputParser()
result = await chain.ainvoke({"question": "..."})
```

Null distribution building is also async:

```python
null_dist = await NullDistribution.afrom_builtin_corpus(
    vectorstore=vectorstore,
    embeddings=embeddings,
    cache_path="./null_dist.npy",
)
```

---

## Evaluation & Benchmarking

Reproduce the research results or benchmark on your own dataset:

```python
from ragnaros.evaluation import EvaluationHarness, RunConfig

async def answer_fn(question: str, docs: list) -> str:
    context = "\n\n".join(d.page_content for d in docs)
    result = await chain.ainvoke({"question": question, "context": context})
    return result["answer"]

harness = EvaluationHarness(
    questions=dataset["question"],
    ground_truths=dataset["answer"],
    answer_fn=answer_fn,
    embeddings=embeddings,
    null_distribution=null_dist,
    vectorstore=vectorstore,
    config=RunConfig(n_questions=100, seed=42),
)

results = harness.run(
    fixed_k_values=[1, 5, 7, 10, 20],
    estimator_names=["higher_criticism", "benjamini_hochberg", "bonferroni"],
)

for r in results:
    print(r.summary())
```

---

## Visualization

```python
from ragnaros.visualization import cost_accuracy_plot, k_distribution_plot, null_vs_real_plot

# Cost vs Accuracy scatter
fig = cost_accuracy_plot(results)
fig.savefig("cost_vs_accuracy.png", dpi=150)

# k distribution per dynamic method
fig = k_distribution_plot(results)

# Null distribution vs real similarities
import numpy as np
real_sims = np.array([...])  # cosine similarities from a real query batch
fig = null_vs_real_plot(null_dist, real_sims)
```

Requires `pip install "ragnaros[viz]"`.

---

## Compatibility

| Component | Supported |
|---|---|
| Vector stores | Any LangChain `VectorStore` (Chroma, FAISS, Pinecone, Qdrant, Weaviate, …) |
| Embedding models | Any LangChain `Embeddings` (OpenAI, HuggingFace, Cohere, Sentence Transformers, …) |
| LLMs | Any LangChain `BaseChatModel` or LCEL chain |
| Python | 3.11+ |
| Async | Full `asyncio` support |

---

## Development

```bash
git clone https://github.com/yarinshitrit/RAGnaros
cd RAGnaros
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=ragnaros --cov-report=term-missing

# Lint
ruff check ragnaros tests

# Type check
mypy ragnaros
```

---

## Research Background

This library is derived from research conducted during the Language Models Seminar at Reichman University (M.Sc. Computer Science, 2025).

**Full title**: *Toward Optimal Retrieval: Dynamic Document Retrieval in Vector-Based Search*
**Author**: Yarin Shitrit

The research evaluates three statistical multiple-testing corrections adapted for the k-selection problem:

- **Bonferroni** (Bonferroni, 1936): controls FWER — most conservative.
- **Benjamini-Hochberg** (Benjamini & Hochberg, 1995): controls FDR — balanced.
- **Higher Criticism** (Donoho & Jin, 2004): detects sparse, weak signals — best empirical performance.

The null distribution methodology is inspired by the concept of testing query-document relevance against a background of unrelated query-document pairs, making the statistical threshold adaptive to the corpus.

---

## License

MIT © Yarin Shitrit
