import streamlit as st
import os
import time
import pandas as pd
from pypdf import PdfReader
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# --- 1. CLOUD SECURITY ---
if "GOOGLE_API_KEY" in st.secrets and "PINECONE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
else:
    st.error("❌ API Keys Missing! Add GOOGLE_API_KEY and PINECONE_API_KEY to Streamlit Secrets.")
    st.stop()

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="ValdezAI Vault", page_icon="🛡️", layout="wide")
st.title("🛡️ ValdezAI Private Intelligence Vault")

# --- 3. PERSISTENT SYSTEM CORE ---
index_name = "valdezai-vault"

# Initialize 2026 Stable Embeddings (FORCED 768 DIMENSIONS)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_document",
    output_dimensionality=768
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def process_pdf(pdf_file):
    """Rips text and vaults into Pinecone with a safety throttle."""
    reader = PdfReader(pdf_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    text = text.encode("utf-8", "ignore").decode("utf-8").replace('\x00', '')
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = text_splitter.split_text(text)
    
    with st.spinner(f"🔒 Vaulting {len(chunks)} chunks into Pinecone..."):
        for i in range(0, len(chunks), 5):
            batch = chunks[i:i+5]
            # This directly sends the PDF data to your Pinecone cloud
            PineconeVectorStore.from_texts(
                batch, 
                embeddings, 
                index_name=index_name,
                namespace="default_user"
            )
            time.sleep(1) # Safe throttle for Free Tier
    st.success("Vault Updated!")

# --- 4. SESSION MANAGEMENT ---
if "messages" not in st.session_state: st.session_state.messages = []

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("📂 Data Management")
    uploaded_file = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            with open("temp_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("CSV Ready")
        elif uploaded_file.name.endswith('.pdf'):
            process_pdf(uploaded_file)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 6. CHAT INTERFACE (SMART SEARCH) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask ValdezAI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ""
        try:
            # STEP 1: Always check the PDF Vault (Pinecone) first
            vector_db = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace="default_user")
            qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_db.as_retriever())
            
            pdf_response = qa_chain.run(prompt)
            
            # If Pinecone has a valid answer, use it
            if "I don't know" not in pdf_response and "not mentioned" not in pdf_response and "does not contain" not in pdf_response:
                response = pdf_response
            
            # STEP 2: Fallback to CSV if PDF has no answer
            if not response:
                path = "temp_data.csv" if os.path.exists("temp_data.csv") else "commission_data.csv"
                if os.path.exists(path):
                    agent = create_csv_agent(llm, path, allow_dangerous_code=True)
                    response = agent.run(prompt)
                else:
                    response = "I couldn't find an answer in the Vault or CSV. Please upload more data."

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Analysis Error: {e}")




