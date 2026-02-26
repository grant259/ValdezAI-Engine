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

# --- 3. DATA PROCESSING FUNCTIONS ---
def process_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    
    # Cleaning text for the 2026 Embedding Engine
    text = text.replace('\x00', '') # Remove null characters
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    # NEW 2026 MODEL ID: text-embedding-004
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# --- 4. SIDEBAR & FILE UPLOADER ---
with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])
    
    default_csv = "commission_data.csv"
    active_mode = None

    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            with open("temp_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            active_mode = "CSV"
        elif uploaded_file.name.endswith('.pdf'):
            st.session_state.vector_db = process_pdf(uploaded_file)
            active_mode = "PDF"
        st.success(f"Loaded: {uploaded_file.name}")
    elif os.path.exists(default_csv):
        active_mode = "CSV"

# --- 5. INITIALIZE AI ---
# Stable 2026 Brain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- 6. CHAT ENGINE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if active_mode == "CSV":
                path = "temp_data.csv" if uploaded_file else default_csv
                agent = create_csv_agent(llm, path, allow_dangerous_code=True, handle_parsing_errors=True)
                response = agent.run(prompt)
            elif active_mode == "PDF":
                qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=st.session_state.vector_db.as_retriever())
                response = qa_chain.run(prompt)
            else:
                response = "Please upload a file to begin."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Analysis Error: {e}")
