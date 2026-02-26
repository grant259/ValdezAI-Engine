import streamlit as st
import os
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. CLOUD SECURITY ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ API Key Missing in Secrets!")
    st.stop()

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="ValdezAI Intelligence", page_icon="🛡️", layout="wide")
st.title("🛡️ ValdezAI Private Intelligence")
st.markdown("---")

# --- 3. DATA & AGENT SETUP ---
# Path handling that works on both Windows and Linux (Cloud)
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "commission_data.csv")

@st.cache_resource
def init_agent():
    # Gemini 2.5 Flash is the 2026 industry standard for speed/math
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    return create_csv_agent(
        llm, 
        csv_path, 
        verbose=True, 
        allow_dangerous_code=True,
        handle_parsing_errors=True,
        max_iterations=10
    )

if os.path.exists(csv_path):
    try:
        df_preview = pd.read_csv(csv_path, encoding='utf-8-sig')
        st.sidebar.success("✅ Records Active")
        st.sidebar.dataframe(df_preview.head(3))
        agent = init_agent()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.error("❌ Database not found. Ensure CSV is in the same folder as this script.")
    st.stop()

# --- 4. THE CONVERSATIONAL ENGINE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "ValdezAI is online. How can I help you analyze your commissions?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question or say hello..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # ROUTING LOGIC: Handle social cues without breaking the Agent
        social_cues = ["thank", "great job", "good work", "hello", "hi", "awesome", "help", "who are you"]
        
        if any(word in prompt.lower() for word in social_cues):
            if "help" in prompt.lower() or "who are you" in prompt.lower():
                response = "I am ValdezAI. I can calculate averages, find specific job amounts, or tell you who your top performers are based on your uploaded CSV."
            else:
                response = "You're very welcome! I'm glad the analysis was helpful. What's our next task?"
            st.markdown(response)
        else:
            # DATA LOGIC: Use the Agent for math and searching
            with st.spinner("ValdezAI is analyzing..."):
                try:
                    response = agent.run(prompt)
                    st.markdown(response)
                except Exception as e:
                    # Catch the 'Output Parsing' error but show the AI's actual message
                    if "Could not parse LLM output" in str(e):
                        response = str(e).split("Could not parse LLM output:")[1].replace("`", "").strip()
                        st.markdown(response)
                    else:
                        st.error(f"Analysis Error: {e}")
                        response = "I had trouble calculating that. Could you rephrase the question?"

    st.session_state.messages.append({"role": "assistant", "content": response})

