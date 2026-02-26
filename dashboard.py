import streamlit as st
import os
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. CLOUD SECURITY SETUP ---
# This pulls your API key from the "Advanced Settings > Secrets" box you just filled out
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ GOOGLE_API_KEY not found in Streamlit Secrets. Please add it in Advanced Settings.")
    st.stop()

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="ValdezAI Cloud Engine", page_icon="🛡️", layout="wide")
st.title("🛡️ ValdezAI Private Intelligence")
st.markdown("---")

# --- 3. DATA LOADING ---
# Path on GitHub should match your repository structure
csv_path = "docs_to_index/commission_data.csv"

@st.cache_resource
def init_agent():
    # Using Gemini 1.5 Flash: It's the 'Jet Engine' for this logic
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Create the 'Fuzzy Logic' Agent
    agent = create_csv_agent(
        llm, 
        csv_path, 
        verbose=True, 
        allow_dangerous_code=True, # Allows the AI to do math/calculations
        handle_parsing_errors=True
    )
    return agent

# Verify file exists on GitHub before starting
if os.path.exists(csv_path):
    # Verify the CSV encoding is clean for the cloud
    try:
        # Show a small preview in the sidebar so you know it's working
        df_preview = pd.read_csv(csv_path, encoding='utf-8-sig')
        st.sidebar.success("✅ Records Online")
        st.sidebar.dataframe(df_preview.head(3))
    except:
        st.sidebar.error("⚠️ CSV Encoding issue. Ensure it is saved as 'CSV (Comma delimited)'.")
    
    agent = init_agent()
else:
    st.error(f"❌ Database not found at {csv_path}. Check your GitHub folder structure.")
    st.stop()

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "ValdezAI Cloud is active. How can I help you analyze your data?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about names, averages, or totals..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("ValdezAI is thinking..."):
            try:
                # The Cloud Agent writes code, runs math, and gives the answer
                response = agent.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Agent Error: {e}")
