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
    output_dimensionality=768 # <--- The 2026 Redacted Error Fix
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def process_pdf(pdf_file):
    """Rips text and vaults into Pinecone with a safety throttle."""
    reader = PdfReader(pdf_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    text = text.encode("utf-8", "ignore").decode("utf-8").replace('\x00', '')
    
    # Smaller chunks help avoid Google's 2026 per-request token limits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = text_splitter.split_text(text)
    
    with st.spinner(f"🔒 Vaulting {len(chunks)} chunks into Pinecone..."):
        # We process in small batches of 5 to avoid 429 Quota errors
        for i in range(0, len(chunks), 5):
            batch = chunks[i:i+5]
            vector_store = PineconeVectorStore.from_texts(
                batch, 
                embeddings, 
                index_name=index_name,
                namespace="default_user"
            )
            time.sleep(2) # Safe throttle for Free Tier
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
            st.success("CSV Ready")
        elif uploaded_file.name.endswith('.pdf'):
            try:
                process_pdf(uploaded_file)
                st.session_state.active_mode = "PDF"
                st.success("Vault Updated")
            except Exception as e:
                st.error(f"Vaulting Failed: {e}")

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
                # Connect to existing cloud vault
                vector_db = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace="default_user")
                qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_db.as_retriever())
                response = qa_chain.run(prompt)
            else:
                response = "Vault is empty. Please upload a file."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Analysis Error: {e}")





