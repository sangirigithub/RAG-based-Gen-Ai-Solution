SmartVaultHub: RAG-Based Question Answering Solution

Overview
This project implements a Retrieval-Augmented Generation (RAG) Question Answering system leveraging open-source embeddings, LLMs and vector stores. It supports end-to-end pipeline steps from dataset ingestion (Stanford SQuAD v2) to generating accurate answers based on retrieved context.
________________________________________________________________________________
Features
•	Load and preprocess SQuAD v2 dataset
•	Chunk large documents for efficient retrieval
•	Generate Embeddings using sentence-transformers/all-MiniLM-L6-v2 with LangChain and Chroma vector store
•	Perform Similarity search to retrieve top-k relevant context documents
•	Generate answers using configurable LLM models (Groq, Ollama, HuggingFace)
•	Evaluate model outputs using Exact Match scoring
•	Interactive web UI with Streamlit for user-friendly querying
________________________________________________________________________________
Setup
Prerequisites
•	Python 3.8+
•	Install dependencies:

pip install -r requirements.txt
•	Obtain API keys for LLM models (e.g., GroqCloud, Ollama, HuggingFace) and add them to .env
________________________________________________________________________________
Usage
Data preparation
Run the data download script to fetch and store SQuAD dataset:
python download.py
Embedding generation and ingestion
Run the ingestion pipeline to parse data, chunk text, and generate embeddings:
python ingest.py  # followed by embeddings generation depending on your setup

Run the Streamlit app
streamlit run QnA_RAG.py

Use the sidebar to select LLM, input your API key, choose top-k retrieval docs, and type your question.

Create API key and access available Open-source Models by visiting:

GroqCloud API Keys Console: https://console.groq.com/keys
Ollama Model Library: https://ollama.com/library
HuggingFace Models: https://huggingface.co/settings/tokens
________________________________________________________________________________
File Structure
•	download.py — Download SQuAD dataset and save CSV files
•	ingest.py — Load, parse, chunk dataset for retrieval
•	embeddings.py — Generate and persist vector embeddings
•	retriever.py — Retrieve relevant context documents using semantic search
•	generator.py — Generate answers using LLMs with retrieved context
•	prompt.py — Craft prompt templates for LLM inputs
•	evaluator.py — Evaluate generated answers using Exact Match
•	QnA_RAG.py — Main pipeline integrating components with Streamlit UI
________________________________________________________________________________
Evaluation
This project uses Exact Match (EM) metric to evaluate how many generated answers exactly match reference answers from the dataset for reliable assessment.
________________________________________________________________________________
Notes
•	Ensure proper API keys are set in .env for LLM access
•	You may customize embedding models and retrieval configurations in the code/config
•	The system is modular allowing swapping or upgrading components
________________________________________________________________________________


