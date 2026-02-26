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
    st.error("❌ API Key Missing!")
    st.stop()

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="ValdezAI Universal", page_icon="🛡️", layout="wide")
st.title("🛡️ ValdezAI Private Intelligence")

# --- 3. PERSISTENT DATA PROCESSING ---
def process_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    # Join text and remove null characters that crash 2026 models
    text = "".join([page.extract_text() or "" for page in reader.pages]).replace('\x00', '')
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    # 2026 STABLE MODEL: gemini-embedding-001
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT" # Critical for 2026 API stability
    )
    
    # Build the 'Vector Brain'
    return FAISS.from_texts(chunks, embeddings)
# Initialize Session States
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "active_mode" not in st.session_state: st.session_state.active_mode = "CSV"

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])
    
    if uploaded_file:
        # If user uploads a NEW file, clear the old brain to prevent math errors
        if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.vector_db = None
            st.session_state.last_file = uploaded_file.name

        if uploaded_file.name.endswith('.csv'):
            with open("temp_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.active_mode = "CSV"
        elif uploaded_file.name.endswith('.pdf'):
            if st.session_state.vector_db is None:
                with st.spinner("Building PDF Brain with Gemini Embeddings..."):
                    st.session_state.vector_db = process_pdf(uploaded_file)
            st.session_state.active_mode = "PDF"
        st.success(f"Active: {uploaded_file.name}")

# --- 5. INITIALIZE AI ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Display Chat History
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
                response = "I'm ready. Please upload a file to begin analysis."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Analysis Error: {e}")


