# ragas_evaluation.py
# Demonstrates RAG evaluation using RAGAS with local Ollama (llama3.2:3b)
#
# ragas 0.4.x uses singleton metrics from ragas.metrics (not .collections)
# Ollama's OpenAI-compatible endpoint lets us reuse OpenAI-based clients

import warnings
warnings.filterwarnings("ignore")

from datasets import Dataset
from langchain_openai import ChatOpenAI

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

# ── Ollama via OpenAI-compatible API ─────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434/v1"
LLM_MODEL        = "llama3.2:3b"
EMBEDDING_MODEL  = "nomic-embed-text"   # free local embedding model

# LangChain wrappers pointing at Ollama
langchain_llm = ChatOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",               # Ollama ignores the key value
    model=LLM_MODEL,
    temperature=0,
)

# Wrap for ragas
ragas_llm = LangchainLLMWrapper(langchain_llm)

# Inject into singleton metrics (ragas 0.4.x pattern)
faithfulness.llm        = ragas_llm
context_precision.llm   = ragas_llm
context_recall.llm      = ragas_llm

# Increase timeout — llama3.2:3b is slower than cloud LLMs
run_config = RunConfig(timeout=240, max_retries=2)

# ── Sample data ───────────────────────────────────────────────────────────────
dataset = Dataset.from_dict({
    "question": [
        "What documents are required for a home loan?"
    ],
    "contexts": [
        [
            "Home loan applicants need identity proof and address proof.",
            "Applicants should submit salary slips and bank statements.",
        ]
    ],
    "answer": [
        "Applicants need identity proof, address proof, "
        "salary slips and bank statements."
    ],
    "ground_truth": [
        "Applicants need identity proof, address proof, "
        "salary slips and bank statements."
    ],
})

# ── Evaluate ──────────────────────────────────────────────────────────────────
result = evaluate(
    dataset,
    metrics=[faithfulness, context_precision, context_recall],
    run_config=run_config,
)

print("=" * 60)
print(f"RAGAS EVALUATION  (model: {LLM_MODEL})")
print("=" * 60)
print(result)
