import streamlit as st
import os
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. CLOUD SECURITY & SECRETS ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ API Key Missing: Go to Streamlit Advanced Settings > Secrets and add GOOGLE_API_KEY = 'your_key'")
    st.stop()

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="ValdezAI Cloud Engine", page_icon="🛡️", layout="wide")
st.title("🛡️ ValdezAI Private Intelligence")
st.markdown("---")

# --- 3. DYNAMIC DATA LOADING ---
# This looks for the CSV in the same folder as the script automatically
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "commission_data.csv")

@st.cache_resource
def init_agent():
    # UPDATE: Using the latest stable 2026 model ID
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    agent = create_csv_agent(
        llm, 
        csv_path, 
        verbose=True, 
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        max_iterations=10 # Gives it more room to think
    )
    return agent

if os.path.exists(csv_path):
    try:
        # Sidebar preview ensures the file is 'readable'
        df_preview = pd.read_csv(csv_path, encoding='utf-8-sig')
        st.sidebar.success("✅ Records Online")
        st.sidebar.dataframe(df_preview.head(5))
        agent = init_agent()
    except Exception as e:
        st.sidebar.error(f"⚠️ CSV Error: {e}")
        st.stop()
else:
    st.error(f"❌ Database not found. Ensure 'commission_data.csv' is in the same folder as this script.")
    st.info(f"Currently looking at: {csv_path}")
    st.stop()

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "ValdezAI Cloud is active. Ask me about your commission records."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ex: 'Who had the highest commission?' or 'What is the average?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("ValdezAI is calculating..."):
            try:
                # The Gemini Agent reads the CSV and generates a response
                response = agent.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Agent Error: {e}")
