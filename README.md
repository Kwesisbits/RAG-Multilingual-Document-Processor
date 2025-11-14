---
title: RAG Multilingual Document Processor
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
---
# 🤖 RAG Multilingual Document Processor

An AI-powered document Q&A and translation system supporting 12+ languages.

## Features

- **Context-Aware Q&A**: Ask questions about your documents
- **Full Document Translation**: Translate PDFs into multiple languages
- **12+ Languages**: English, French, Spanish, German, Chinese, Japanese, and more
- **RAG Pipeline**: Retrieval-Augmented Generation for accurate answers

## How It Works

1. Upload a PDF document
2. Choose between Q&A or translation mode
3. Select your target language
4. Get AI-powered responses

## Technology Stack

- **LangChain**: RAG pipeline orchestration
- **HuggingFace Transformers**: Flan-T5 (QA) + mBART (Translation)
- **ChromaDB**: Vector storage
- **Gradio**: Web interface git clone 
