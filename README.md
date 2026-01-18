MEDICARECHAT - RAG-based Medical Chatbot 🏥
MEDICARECHAT is an intelligent AI chatbot designed to provide accurate answers based on medical PDF data. It utilizes a Retrieval-Augmented Generation (RAG) architecture to ensure that responses are context-aware and grounded in the provided medical documents.

🚀 Features
PDF Knowledge Integration: Automatically processes medical documents to build a local knowledge base.

Fast Vector Search: Uses FAISS for high-speed similarity searches across medical data.

Agentic Capabilities: Built with LangGraph to handle complex, multi-step medical queries.

Memory Retention: Maintains conversation history for a natural, chat-like experience.

🛠️ Tech Stack
Language: Python

Orchestration: LangChain & LangGraph

Vector Database: FAISS

API Framework: FastAPI

Package Management: UV

📂 Project Structure
medibot.py: The main script to run the chatbot interface.

create_memory_for_llm.py: A script to generate embeddings and initialize the vector store.

connect_memory_with_llm.py: Handles the connection between the LLM and the FAISS database.

data/: Directory for storing your medical source PDFs.

vectorstore/db_faiss/: Stores the indexed vector data for retrieval.

⚙️ How to Run
Install Dependencies:

Bash

pip install -r requirements.txt
Initialize Knowledge Base:

Bash

python create_memory_for_llm.py
Launch the Chatbot:

Bash

python medibot.py
📝 Disclaimer
This project is for educational and portfolio purposes only. Always consult a certified healthcare professional for actual medical advice.
