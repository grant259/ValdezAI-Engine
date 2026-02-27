import streamlit as st
import os
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
    st.error("❌ API Keys Missing! Please add them to Streamlit Secrets.")
    st.stop()

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="ValdezAI Vault", page_icon="🛡️", layout="wide")
st.title("🛡️ ValdezAI Private Intelligence Vault")

# --- 3. PERSISTENT SYSTEM CORE ---
# We initialize these at the top level so the app 'remembers' them on every rerun
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
index_name = "valdezai-vault"

@st.cache_resource
def get_vector_store():
    """Connects to the permanent cloud vault."""
    return PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace="default_user")

def process_pdf(pdf_file):
    """Rips text from PDF and vaults it into Pinecone."""
    reader = PdfReader(pdf_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    # Clean hidden characters that crash 2026 APIs
    text = text.encode("utf-8", "ignore").decode("utf-8").replace('\x00', '')
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    with st.spinner("🔒 Vaulting records into Pinecone..."):
        vector_store = get_vector_store()
        # Direct upsert - bypasses the from_texts buggy helper
        vector_store.add_texts(chunks)
    return vector_store

# --- 4. SESSION MANAGEMENT ---
if "messages" not in st.session_state: st.session_state.messages = []
if "active_mode" not in st.session_state: st.session_state.active_mode = "CSV"

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("📂 Data Management")
    uploaded_file = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            with open("temp_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.active_mode = "CSV"
            st.success("CSV Loaded")
        elif uploaded_file.name.endswith('.pdf'):
            process_pdf(uploaded_file)
            st.session_state.active_mode = "PDF"
            st.success("PDF Vaulted Successfully")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 6. CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask ValdezAI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if st.session_state.active_mode == "CSV":
                path = "temp_data.csv" if os.path.exists("temp_data.csv") else "commission_data.csv"
                agent = create_csv_agent(llm, path, allow_dangerous_code=True, handle_parsing_errors=True)
                response = agent.run(prompt)
            elif st.session_state.active_mode == "PDF":
                vector_db = get_vector_store()
                qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_db.as_retriever())
                response = qa_chain.run(prompt)
            else:
                response = "I'm ready. Please upload a file to begin."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Vault Analysis Error: {e}")




