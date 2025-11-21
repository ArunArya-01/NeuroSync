import streamlit as st
import os
from dotenv import load_dotenv
from typing import TypedDict
import PyPDF2
from streamlit_mic_recorder import speech_to_text
import pandas as pd
import json

# AI Libraries
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="NeuroSync OS", page_icon="🧠", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    /* Remove extra padding at top */
    .block-container {
        padding-top: 2rem;
    }
    /* Style the metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
        color: #4F46E5;
    }
    /* Chat input styling */
    .stChatInput {
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.error("🚨 GROQ_API_KEY missing in .env file.")
    st.stop()

# --- 2. MODELS ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_key)

# --- 3. FUNCTIONS ---
def extract_profile_from_pdf(text):
    prompt = f"""
    Extract student details from this text. Return JSON ONLY: 
    {{ "name": "Name", "diagnosis": "Diagnosis", "grade": "Grade", "iep_date": "Date" }}
    TEXT: {text[:3000]}
    """
    try:
        res = llm.invoke(prompt).content.replace("```json", "").replace("```", "").strip()
        return json.loads(res)
    except:
        return {"name": "Unknown", "diagnosis": "Not Found", "grade": "N/A", "iep_date": "N/A"}

# --- 4. AGENT GRAPH ---
class AgentState(TypedDict):
    user_request: str
    router_decision: str
    final_response: str
    active_agent: str

def router_node(state):
    prompt = f"Classify '{state['user_request']}' into: 'compliance', 'history', 'strategy', 'analytics'. Return 1 word."
    try: decision = llm.invoke(prompt).content.strip().lower()
    except: decision = "strategy"
    
    if "compliance" in decision: return {"router_decision": "compliance"}
    if "history" in decision: return {"router_decision": "history"}
    if "analytics" in decision: return {"router_decision": "analytics"}
    return {"router_decision": "strategy"}

def compliance_agent(state):
    return {"final_response": llm.invoke(f"Check IDEA Act compliance: {state['user_request']}").content, "active_agent": "Compliance Agent"}

def history_agent(state):
    pdf = st.session_state.get("pdf_context", "")
    if not pdf: return {"final_response": "📂 Please upload a PDF in the Sidebar first.", "active_agent": "System"}
    return {"final_response": llm.invoke(f"Context: {pdf[:15000]}. Answer: {state['user_request']}").content, "active_agent": "History Agent"}

def strategy_agent(state):
    return {"final_response": llm.invoke(f"Create teaching strategy: {state['user_request']}").content, "active_agent": "Strategy Agent"}

def analytics_agent(state):
    return {"final_response": "I've updated the Behavioral Dashboard with this incident.", "active_agent": "Analytics Agent"}

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("compliance", compliance_agent)
workflow.add_node("history", history_agent)
workflow.add_node("strategy", strategy_agent)
workflow.add_node("analytics", analytics_agent)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x['router_decision'], 
    {"compliance": "compliance", "history": "history", "strategy": "strategy", "analytics": "analytics"})
workflow.add_edge("compliance", END)
workflow.add_edge("history", END)
workflow.add_edge("strategy", END)
workflow.add_edge("analytics", END)
app_graph = workflow.compile()

# --- 5. UI LAYOUT ---

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "NeuroSync Online. Ready.", "agent": "System"}]
if "profile" not in st.session_state:
    st.session_state.profile = {"name": "Waiting for PDF...", "diagnosis": "---", "grade": "---", "iep_date": "---"}

# --- SIDEBAR (CONTROLS) ---
with st.sidebar:
    st.title("📂 Case File")
    
    # 1. PDF Uploader
    uploaded_file = st.file_uploader("Upload IEP (PDF)", type="pdf")
    if uploaded_file:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join([page.extract_text() for page in reader.pages])
            st.session_state["pdf_context"] = text
            
            # Auto-Extract Profile (Only once)
            if st.session_state.profile["name"] == "Waiting for PDF...":
                with st.spinner("Analyzing PDF..."):
                    st.session_state.profile = extract_profile_from_pdf(text)
            st.toast("PDF Loaded Successfully!", icon="✅")
        except: st.error("PDF Error")

    # 2. Extracted Profile Card
    with st.container(border=True):
        st.subheader(st.session_state.profile["name"])
        st.caption(f"Diagnosis: {st.session_state.profile['diagnosis']}")
        c1, c2 = st.columns(2)
        c1.metric("Grade", st.session_state.profile["grade"])
        c2.metric("IEP Due", st.session_state.profile["iep_date"])

    # 3. Voice Input (Moved here for clean layout)
    st.divider()
    st.write("🎙️ **Voice Command**")
    audio_text = speech_to_text(language='en', just_once=True, key='mic_sidebar')
    
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN PAGE ---
st.title("🧠 NeuroSync OS")

tab1, tab2 = st.tabs(["💬 Case Assistant", "📊 Analytics Dashboard"])

# --- TAB 1: CHAT ---
with tab1:
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "agent" in msg and msg["agent"] != "User":
                st.caption(f"⚡ {msg['agent']}")
            st.write(msg["content"])

    # Handle Inputs (Voice OR Text)
    # We check Voice first from Sidebar
    final_input = None
    
    if audio_text:
        final_input = audio_text
    
    # Standard Chat Input (Bottom, Full Width)
    if prompt := st.chat_input("Type your request here..."):
        final_input = prompt

    # Process Input
    if final_input:
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": final_input, "agent": "User"})
        with st.chat_message("user"):
            st.write(final_input)

        # Run AI
        with st.chat_message("assistant"):
            with st.spinner("Agents working..."):
                result = app_graph.invoke({"user_request": final_input})
                response = result['final_response']
                agent = result['active_agent']
                
                st.caption(f"⚡ {agent}")
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response, "agent": agent})

# --- TAB 2: ANALYTICS ---
with tab2:
    st.header("Behavioral Trends")
    
    # Mock Data
    data = {"Day": ["M", "T", "W", "Th", "F"], "Incidents": [3, 1, 5, 2, 4], "Focus %": [40, 70, 30, 80, 50]}
    df = pd.DataFrame(data)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Incidents")
        st.bar_chart(df.set_index("Day")["Incidents"], color="#ff4b4b")
    with c2:
        st.subheader("Focus Duration")
        st.line_chart(df.set_index("Day")["Focus %"], color="#4F46E5")