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

# Initialize Pinecone Client
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index_name = "valdezai-vault" # Ensure this matches your Pinecone Index name

# --- 3. DATA PROCESSING ---
def process_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "".join([page.extract_text() or "" for page in reader.pages]).replace('\x00', '')
    
    # Split into business-sized chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Send to Pinecone Cloud
    # NOTE: 'namespace' allows you to keep different clients' data separate
    with st.spinner("🔒 Vaulting data into permanent cloud storage..."):
        vector_store = PineconeVectorStore.from_texts(
            chunks, 
            embeddings, 
            index_name=index_name,
            namespace="default_user" 
        )
    return vector_store

# State Management
if "messages" not in st.session_state: st.session_state.messages = []
if "active_mode" not in st.session_state: st.session_state.active_mode = "CSV"

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📂 Data Management")
    uploaded_file = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            with open("temp_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.active_mode = "CSV"
        elif uploaded_file.name.endswith('.pdf'):
            process_pdf(uploaded_file)
            st.session_state.active_mode = "PDF"
            st.success(f"✅ {uploaded_file.name} is now in the Vault.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 5. INITIALIZE AI ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Connect to existing vault if it exists
try:
    vector_db = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace="default_user")
except Exception:
    vector_db = None

# --- 6. CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your vaulted records..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if st.session_state.active_mode == "CSV":
                path = "temp_data.csv" if os.path.exists("temp_data.csv") else "commission_data.csv"
                agent = create_csv_agent(llm, path, allow_dangerous_code=True, handle_parsing_errors=True)
                response = agent.run(prompt)
            elif st.session_state.active_mode == "PDF" and vector_db:
                qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_db.as_retriever())
                response = qa_chain.run(prompt)
            else:
                response = "The Vault is empty. Please upload a file to begin."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Vault Error: {e}")




