import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Local environment variables ko load karne ke liye
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    # Exact wahi model jo aapne create_memory mein use kiya tha (Bina kisi API token ke)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def main():
    st.set_page_config(page_title="MediBot", page_icon="🤖", layout="centered")
    st.title("🤖 MediBot: Ask Your Health Assistant!")
    st.write("Welcome! Ask anything based on the pre-loaded medical data.")

    # Session state initialize karna chat history ke liye
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Purane messages ko UI par display karna
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # User input chat box
    user_prompt = st.chat_input("Pass your prompt here...")

    if user_prompt:
        # User ka message screen par dikhana aur history mein save karna
        with st.chat_message('user'):
            st.markdown(user_prompt)
        st.session_state.messages.append({'role': 'user', 'content': user_prompt})
                
        try: 
            # Vector store load karna
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # API Keys fetch karna
            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            GROQ_MODEL_NAME = "llama-3.1-8b-instant"
            
            if not GROQ_API_KEY:
                st.error("⚠️ GROQ_API_KEY nahi mili! Please check environment variables.")
                return

            # LLM Configuration
            llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=0.5,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )
            
            # Retriever setup
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            
            # RAG Prompt template
            retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the user's question using only the provided context. If you do not know the answer, say that you do not know.\n\nContext:\n{context}"),
                ("human", "{input}")
            ])

            # Chunks ko merge karne ka function
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # RAG Chain Creation (LCEL syntax)
            rag_chain = (
                {
                    "context": retriever | format_docs, 
                    "input": RunnablePassthrough()
                }
                | retrieval_qa_chat_prompt 
                | llm 
                | StrOutputParser()
            )

            # Assistant ka response generate aur display karna
            with st.chat_message('assistant'):
                with st.spinner("Thinking..."):
                    result = rag_chain.invoke(user_prompt)
                    st.markdown(result)
            
            # Assistant ka message history mein save karna
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()