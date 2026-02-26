import streamlit as st
import os
import pandas as pd
from pypdf import PdfReader
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

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
        text += page.extract_text()
    
    # Split text into chunks so the AI can 'digest' it
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    # Create a searchable 'Vector Brain'
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# --- 4. SIDEBAR & FILE UPLOADER ---
with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV (Data) or PDF (Manuals/Contracts)", type=["csv", "pdf"])
    
    # Default file if nothing is uploaded
    default_csv = "commission_data.csv"
    active_mode = None

    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            with open("temp_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            active_mode = "CSV"
        elif uploaded_file.name.endswith('.pdf'):
            vector_db = process_pdf(uploaded_file)
            active_mode = "PDF"
        st.success(f"Loaded: {uploaded_file.name}")
    elif os.path.exists(default_csv):
        active_mode = "CSV"
        csv_path = default_csv

# --- 5. INITIALIZE AI ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- 6. CHAT ENGINE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your data or document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if active_mode == "CSV":
                # Use the CSV Agent for math/records
                agent = create_csv_agent(llm, "temp_data.csv" if uploaded_file else default_csv, 
                                         allow_dangerous_code=True, handle_parsing_errors=True)
                response = agent.run(prompt)
            elif active_mode == "PDF":
                # Use RAG for reading contracts/manuals
                qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_db.as_retriever())
                response = qa_chain.run(prompt)
            else:
                response = "Please upload a file to start the analysis."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")
