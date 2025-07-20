# 📘 Graph RAG Evaluation Tool

A **modular Python tool** for evaluating the **recall and precision of facts** in a summary compared to the source text using **Graph RAG**, **NLP**, and **LLM-based fact extraction**.

---

## 🧠 Features

- ✅ Multiple Fact Extraction Methods:
  - `spaCy` – Rule-based extraction
  - `Ollama LLM` (e.g., `llama3`, `mistral`) – LLM-based extraction
  - `Hybrid` – Combines both for better coverage

- 🔍 Fuzzy Matching of Relations:
  - Uses **Sentence Transformers** (`all-MiniLM-L6-v2`) to compare semantic similarity between relations

- 📊 Graph Visualization:
  - `NetworkX` for in-memory graph visualization
  - `Neo4j` for large-scale knowledge graph storage and querying

- 📄 Chunking Support:
  - Uses `textwrap` to handle long documents

- 🖥️ CLI Interface:
  - Easy-to-use command-line interface for testing and evaluation

---

## 📁 Folder Structure
graph_rag_eval/
│
├── config.yaml
├── requirements.txt
│
├── fact_extractor/ # Fact extraction logic
├── graph_builder/ # Build NetworkX or Neo4j graphs
├── graph_comparer/ # Compare facts using embeddings
├── visualizer/ # Visualize graphs
├── chunker/ # Chunk long texts
├── cli/ # CLI interface
├── main.py # Sample runner
└── sample_inputs/ # Test input texts


---

## 🧪 Sample Input Texts

- `sample1`: Short text about Barack Obama
- `sample2`: Medium text about Elon Musk
- `sample3`: Long text about Artificial Intelligence

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/graph-rag-eval.git 
cd graph-rag-eval
pip install -r requirements.txt
```

## CLI Options
--source TEXT               Enter source text
--summary TEXT              Enter summary text
--method [spacy|ollama|hybrid]
--model TEXT                Ollama model name (default: llama3)
--threshold FLOAT           Similarity threshold for fuzzy match (default: 0.75)
--chunk / --no-chunk        Enable chunking for long texts
--store / --no-store        Store graphs in Neo4j