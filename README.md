# 🏥 MEDICARECHAT: RAG-Driven Medical Chatbot 🤖

An intelligent, context-aware AI assistant engineered to extract semantic insights from medical literature and provide grounded, highly accurate responses. By leveraging a Retrieval-Augmented Generation (RAG) architecture, this system mitigates LLM hallucinations and ensures all answers are mathematically tethered to the underlying knowledge base.

🌍 **Live Production Link:** [https://medicarechat.onrender.com](https://medicarechat.onrender.com)

---

## 🚀 Key Features

* **PDF Knowledge Integration:** Automates data ingestion by parsing raw medical PDFs and processing text into dense vector embeddings.
* **Ultra-Fast Vector Storage:** Utilizes **FAISS (Facebook AI Similarity Search)** for high-dimensional, sub-millisecond semantic similarity searches.
* **Advanced Orchestration:** Powered by **LangChain** utilizing the LangChain Expression Language (LCEL) and stateful configurations for optimized pipeline routing.
* **Streamlit Chat Interface:** Built with a premium, stateful UI featuring message history tracking and performance-focused resource caching (`@st.cache_resource`).
* **Production-Ready Containerization:** Standardized environment using a multi-stage **Dockerfile** for deterministic builds on cloud platforms.

---

## 🛠️ Technical Stack

* **Language:** Python 3.10
* **LLM Architecture:** ChatGroq (`llama-3.1-8b-instant`)
* **Embeddings Model:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Store:** FAISS (CPU Optimized)
* **Frontend Framework:** Streamlit (Stateful Chat Elements)
* **Environment Management:** Python-Dotenv & UV / Pipenv

---

## 📂 Architecture & Project Structure

```text
MEDICARECHAT/
├── data/                         # Directory for source medical PDFs
│   └── The_GALE_ENCYCLOPEDIA.pdf
├── vectorstore/
│   └── db_faiss/                 # Persisted high-dimensional vector index
│       ├── index.faiss
│       └── index.pkl
├── medibot.py                    # Main Streamlit Web Application
├── create_memory_for_llm.py      # Knowledge Ingestion & Embedding Pipeline
├── connect_memory_with_llm.py    # Local CLI Evaluation / Testing Script
├── requirements.txt              # Production Python Dependencies
└── Dockerfile                    # Production Container Deployment Config
