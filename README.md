# Vesuvius Discord Study

This project analyzes the Vesuvius Challenge Discord server to uncover discussion themes and sentiment trends through topic modeling and Retrieval-Augmented Generation (RAG). It uses a combination of preprocessing scripts, LDA topic modeling, vector embedding with ChromaDB, and question-answering using language models.

---

## 📁 Repository Structure

```
vesuvius_discord_study/
├── JSON_filter/         # Scripts for filtering and cleaning raw JSON exports
├── LDA/                 # Topic modeling, visualization, and preprocessing
├── RAG/                 # Document chunking, vector storage, and QA via LLMs
```

---

## 🔍 JSON\_filter

### `filter.py`

**Description:**

* Filters all `.json` files in `discordout-2025-01`.
* Keeps messages with timestamps between **March 15, 2023** and **October 28, 2023**.
* Normalizes ISO timestamps and removes invalid entries.
* Outputs cleaned files into `filtered/` with `_filtered.json` suffix.

**How to run:**

```bash
cd JSON_filter
python filter.py
```

**Notes:**

* Make sure the `discordout-2025-01` directory contains the raw JSON exports.
* Filtered files will be saved into a `filtered/` directory.

---

### `move_empty.py`

**Description:**

* Scans all files in `filtered/`.
* Moves any `.json` file where `"messages": []` to an `empty_files/` directory.
* Helps remove empty conversations from downstream analysis.

**How to run:**

```bash
cd JSON_filter
python move_empty.py
```

---

## 🧠 LDA

### `lda.py`

**Description:**

* Loads a specific filtered JSON file.
* Extracts "Default" messages with content.
* Preprocesses text using:

  * Link and special character removal
  * Custom stopword filtering (e.g., "scroll", "ink")
  * Lemmatization via spaCy
* Trains an LDA topic model with Gensim
* Assigns dominant topic to each message
* Outputs:

  * `discord_chat_topics.json` (messages grouped by topic)
  * `lda_visualization.html` (interactive pyLDAvis dashboard)

**How to run:**

```bash
cd LDA
python lda.py
```

**Before running:**

* Update the `json_file` variable with the path to your filtered JSON file.
* Install dependencies:

```bash
pip install spacy nltk gensim pyLDAvis tqdm
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
```

---

### `load_text.py`

**Description:**

* Reads a filtered JSON file and extracts all messages with type "Default" and non-empty content.
* Stores messages in a list (`sample_text`) for quick analysis.
* Prints total number of valid messages.

**How to run:**

```bash
cd LDA
python load_text.py
```

**Notes:**

* Update `json_file` in the script to point to your desired filtered file.
* This script does not write to disk by default but can be easily modified to do so.

---

### `visualize_metadata.py`

**Description:**

* Streamlit app that allows users to upload `discord_chat_topics.json`.
* Visualizes:

  * Top authors per topic using bar charts
  * Message frequency timeline using line charts
* Offers multi-select to focus on specific topics.

**How to run:**

```bash
cd LDA
streamlit run visualize_metadata.py
```

**Requirements:**

```bash
pip install streamlit altair pandas
```

**Notes:**

* Useful for presenting results from `lda.py` interactively.

---

## 🤖 RAG

### `load_messages.py`

**Description:**

* Loads all filtered JSON files in the `JSON_filter/filtered/` directory.
* Sorts messages by timestamp.
* Groups messages into chunks based on time proximity (default: ≤ 1000 seconds).
* Displays histogram of message count per chunk using `matplotlib`.
* Converts message chunks to LangChain `Document` objects with author metadata.
* Embeds and stores documents in a Chroma vector store (`chroma_db_test6`).

**How to run:**

```bash
cd RAG
python load_messages.py
```

**Requirements:**

```bash
pip install matplotlib langchain langchain-chroma langchain-huggingface
```

**Notes:**

* The vector store will be created at `./chroma_db_test6` and persisted.

---

### `rag.ipynb`

**Description:**

* Interactive notebook that sets up a full RAG pipeline using LangGraph and LangChain.
* Components:

  * Loads Chroma vector store (`chroma_db_test6`)
  * Retrieves top-k messages for a query
  * Constructs a prompt template with context
  * Calls Ollama LLM (e.g. `llama3.2`, `deepseek-r1`) to generate an answer
  * Visualizes LangGraph flow with Mermaid diagrams

**How to run:**

```bash
jupyter notebook rag.ipynb
```

**Requirements:**

```bash
pip install langchain langgraph langchain-chroma langchain-huggingface
```

* Ollama must be running with compatible models.

---

### `rag.py`

**Description:**

* Minimal script that demonstrates how to run a similarity search against a Chroma vector store.
* Loads vector store (`chroma_db_test1`) and performs a query using HuggingFace embeddings.
* Prints top 8 documents and their similarity scores.

**How to run:**

```bash
cd RAG
python rag.py
```

**Notes:**

* Update the test query via `phrase = "..."` in the script.
* Ensure the Chroma DB is already populated (e.g., via `load_messages.py`).

**Dependencies:**

```bash
pip install langchain langchain-chroma langchain-huggingface
```

---

## ✅ Summary

This repo provides a complete pipeline to analyze and retrieve insights from Discord conversations using both classical NLP (LDA topic modeling) and modern LLM-based methods (RAG with ChromaDB and Ollama). Each script/module can be run independently or as part of a larger analysis workflow.

Feel free to open issues or submit PRs if you’d like to contribute!
