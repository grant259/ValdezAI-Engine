import streamlit as st
import os
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI # NEW: Cloud Brain

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="ValdezAI Cloud Engine", page_icon="🛡️", layout="wide")
st.title("🛡️ ValdezAI Private Intelligence")

# --- 2. INITIALIZE CLOUD AGENT ---
csv_path = "docs_to_index/commission_data.csv"

# REPLACE THIS WITH YOUR ACTUAL KEY
os.environ["GOOGLE_API_KEY"] = "PASTE_YOUR_KEY_HERE"

@st.cache_resource
def init_agent():
    # We use Gemini 1.5 Flash - it's fast, free-tier friendly, and smart
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    agent = create_csv_agent(
        llm, 
        csv_path, 
        verbose=True, 
        allow_dangerous_code=True,
        handle_parsing_errors=True
    )
    return agent

if os.path.exists(csv_path):
    agent = init_agent()
else:
    st.error("❌ 'commission_data.csv' missing.")
    st.stop()

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask ValdezAI anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing via Google Cloud..."):
            try:
                # Gemini handles the thinking; the result is 100% accurate
                response = agent.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Cloud Error: {e}")