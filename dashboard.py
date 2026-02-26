import streamlit as st
import os
import pandas as pd
from pypdf import PdfReader
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# --- 1. CLOUD SECURITY ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ API Key Missing in Secrets!")
    st.stop()

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="ValdezAI Universal", page_icon="🛡️", layout="wide")
st.title("🛡️ ValdezAI Private Intelligence")

# --- 3. THE "ERROR-KILLER" PDF PROCESSOR ---
def process_pdf(pdf_file):
    # 1. Extract & Clean (Strip weird chars that crash Google's 2026 API)
    reader = PdfReader(pdf_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    text = text.encode("utf-8", "ignore").decode("utf-8").replace('\x00', '')
    
    # 2. Aggressive Small Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    chunks = text_splitter.split_text(text)
    
    # 3. Explicit 2026 Embedding Settings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_document"
    )
    
    # 4. THE FIX: Manually embed in tiny batches to bypass the 'Redacted' Quota Error
    # This avoids the internal LangChain 'batch_embed' logic that is currently failing
    vector_store = FAISS.from_texts(chunks, embeddings, batch_size=5) 
    return vector_store

# State Management
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "active_mode" not in st.session_state: st.session_state.active_mode = "CSV"

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            with open("temp_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.active_mode = "CSV"
        elif uploaded_file.name.endswith('.pdf'):
            with st.spinner("Forcing PDF Brain through Google API..."):
                try:
                    st.session_state.vector_db = process_pdf(uploaded_file)
                    st.session_state.active_mode = "PDF"
                    st.success("PDF Indexed Successfully")
                except Exception as e:
                    st.error(f"Google API Refused: {e}")

# --- 5. INITIALIZE AI ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# --- 6. CHAT ENGINE ---
if prompt := st.chat_input("Ask ValdezAI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if st.session_state.active_mode == "CSV":
                path = "temp_data.csv" if os.path.exists("temp_data.csv") else "commission_data.csv"
                agent = create_csv_agent(llm, path, allow_dangerous_code=True, handle_parsing_errors=True)
                response = agent.run(prompt)
            elif st.session_state.active_mode == "PDF" and st.session_state.vector_db:
                qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=st.session_state.vector_db.as_retriever())
                response = qa_chain.run(prompt)
            else:
                response = "I need a file to analyze. Please upload above."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Analysis Error: {e}")





