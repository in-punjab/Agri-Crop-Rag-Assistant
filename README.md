🌱 Agri Crop Management RAG Assistant

An AI-powered agricultural advisory system built using Retrieval-Augmented Generation (RAG).
The application delivers practical, English-only crop management guidance by combining semantic search with large language model inference.

Live Demo: (https://agri-crop-rag-assistant-parmsangha.streamlit.app/)

📌 Problem Statement

Farmers and agricultural stakeholders often struggle to access accurate, contextual crop management advice quickly. Traditional search methods return scattered results without domain-specific clarity.

This project addresses that challenge by building a domain-focused AI assistant that:

Retrieves relevant agricultural knowledge using semantic vector search

Generates context-aware responses using a large language model

Provides structured, practical crop guidance in real time

🧠 System Architecture

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

User submits a crop-related query

Query is converted into embeddings

FAISS performs similarity search on stored agricultural documents

Top relevant documents are retrieved

Context + user question are passed to the LLM

The model generates practical farming guidance

🏗️ Tech Stack

Frontend:

Streamlit (interactive UI)

Backend / AI:

LangChain

FAISS (Vector Store)

Sentence Transformers (Embeddings)

Groq LLM (LLaMA 3.1 8B Instant)

Infrastructure:

Streamlit Community Cloud (Deployment)

GitHub (Version Control)

✨ Key Features

Retrieval-Augmented Generation (RAG) architecture

Semantic document search using FAISS

English-only controlled AI output

Context-aware agricultural guidance

Source document transparency

Cached model loading for performance optimization

Cloud deployment with secure environment variables

📂 Project Structure
AGRICROP-RAG-ASSISTANT
│
├── data/                # Agricultural knowledge base
├── embeddings/          # FAISS index files
├── src/
│   ├── app.py           # Streamlit application
│   ├── ingest.py        # Data ingestion & embedding creation
│   ├── retriever.py     # Vector store loader
│   └── vector_store.py  # FAISS configuration
│
├── requirements.txt
├── README.md

🚀 Installation (Local Setup)

Clone the repository:

git clone <your-repo-link>
cd AGRICROP-RAG-ASSISTANT


Create virtual environment:

python -m venv venv
venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Create .env file:

GROQ_API_KEY=your_api_key_here


Run the application:

streamlit run src/app.py

☁️ Deployment

The application is deployed on Streamlit Community Cloud.

Deployment Steps:

Push repository to GitHub

Connect repository to Streamlit Cloud

Set main file path: src/app.py

Add GROQ_API_KEY in Streamlit Secrets

📊 Example Use Cases

Pest management guidance

Fertilizer recommendations

Crop disease identification

Irrigation strategy advice

Seasonal crop planning

📈 Future Enhancements

Conversation memory integration

Structured response formatting

Multi-language toggle support

Analytics logging for usage insights

Role-based agricultural advisory modes

🎯 Learning Outcomes

Through this project, the following concepts were implemented and validated:

End-to-end RAG system design

Vector search optimization

Prompt engineering with language control

LLM integration via Groq API

Cloud deployment of AI applications

Session state management in Streamlit
