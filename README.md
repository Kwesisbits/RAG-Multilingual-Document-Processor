# 🤖 RAG Multilingual Document Processor

**A Production-Ready Retrieval-Augmented Generation System for Document Understanding & Translation**

This project implements a **multilingual, retrieval-augmented AI system** capable of performing contextual question answering and full-document translation across 12+ languages. It is designed as a lightweight, modular, and extensible application that can run locally, in HF Spaces, or inside a Dockerized environment.

The system processes unstructured PDF documents, generates embeddings, retrieves contextually relevant chunks, and produces accurate answers or translations using Hugging Face Transformers.

---

## 🚀 Key Capabilities

### 🔍 1. Contextual Document Q&A

Ask natural-language questions about any uploaded document.
The system uses a **RAG (Retrieval-Augmented Generation)** pipeline that retrieves the most relevant text segments before generating an informed answer.

### 🌏 2. Full Document Translation

Translate entire PDF documents into any of 12+ supported languages using **mBART-50**, a multilingual seq-to-seq model specialized for translation tasks.

### 📚 3. Multilingual Support

Currently supports:

| English    | Chinese    |
| French     | Japanese   |
| Spanish    | Arabic     |
| German     | Dutch      |
| Portuguese | Swahili    |
| Italian    | Hindi      |



### 🧠 4. Efficient RAG Pipeline

Built with **LangChain**, the pipeline includes:

* Text extraction + cleaning
* Chunking + preprocessing
* Embedding generation
* ChromaDB vector storage
* Retrieval + LLM inference

---

## 🏗 Project Structure

```
├── app.py                # Main application and Gradio UI
├── requirements.txt      # All dependencies
└── README.md             # Project documentation
```

The application is intentionally kept simple—**only two files**—but the internal logic is highly modular.

---
### 📸 Application Interface
![App Interface Screenshot](App_Interface.png)

## 🔗 Link to app: 
"https://huggingface.co/spaces/KwesiAI/rag-multilingual-processor"

## 🔧 Technology Stack

### Core Components

* **LangChain** – RAG orchestration, document loaders, text splitters, retrieval pipeline
* **HuggingFace Transformers**

  * *Flan-T5*: Generative QA
  * *mBART-50*: Multilingual translation
* **ChromaDB** – Lightweight vector database for persistent retrieval
* **SentencePiece** – Required tokenizer backend for multilingual models
* **PyPDF / pdfplumber** – Robust PDF text extraction
* **Gradio 4.x** – Clean, fast, and interactive UI layer

### Why These Tools?

* **Flan-T5**: Fast, inexpensive, and strong for factual QA
* **mBART-50**: State-of-the-art multilingual translation model
* **ChromaDB**: Embedded vector DB suited for HF Spaces
* **Gradio**: Simple deployment, automatic UI, and HF Spaces compatibility

---

## 🔄 Application Workflow

### 1️⃣ PDF Upload

Users upload a PDF file. The system extracts and preprocesses the text.

### 2️⃣ Mode Selection

* **Document Q&A**
* **Full Translation**

### 3️⃣ Vectorization & Storage (Q&A mode)

* Text is chunked
* Embeddings are generated
* Chunks stored in ChromaDB

### 4️⃣ Context Retrieval

When a question is asked, the retriever returns the most relevant text snippets.

### 5️⃣ LLM Generation

* QA model: Flan-T5
* Translation model: mBART-50

### 6️⃣ Response Delivery

Gradio displays the final answer or translated document output.

---

## 🧑🏽‍💻 Running the App Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
python app.py
```

### 3. Open your browser

The local Gradio URL will appear in your console.

---

## ☁ Deployment

This project runs seamlessly on:

* **HuggingFace Spaces** (recommended)
* **Local machine**

Minimal hardware is required; however, translation models may require additional RAM.

---

## 🧩 Use Cases

* Legal document analysis
* Academic research assistants
* Corporate HR policy search
* Multilingual knowledge extraction
* AI-powered translation hubs
* Document automation workflows

---

## 📌 Requirements

All dependencies are included in **requirements.txt**, including:

* transformers
* sentencepiece
* langchain
* chromadb
* gradio
* pypdf/pdfplumber

---

## 🏁 Summary

This project implements a multilingual RAG system for document Q&A and translation. It demonstrates the integration of retrieval pipelines, large language models, and a lightweight UI for interactive document processing.

## Possible improvements:
* Optimize memory and inference speed for large documents
* Support batch translation and multi-file uploads
* Add user authentication and document management features


