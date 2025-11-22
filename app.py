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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroSync OS", page_icon="🧠", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; color: #4F46E5; }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.error("🚨 GROQ_API_KEY missing.")
    st.stop()

# We use Llama 3.3 70B for all agents (Fast & Smart)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_key)

# --- 2. INTELLIGENT FUNCTIONS ---
def extract_profile_from_pdf(text):
    prompt = f"""
    Extract student details from the text below. Return JSON ONLY: 
    {{ "name": "Name", "diagnosis": "Diagnosis", "grade": "Grade", "iep_date": "Date" }}
    TEXT: {text[:3000]}
    """
    try:
        res = llm.invoke(prompt).content.replace("```json", "").replace("```", "").strip()
        return json.loads(res)
    except:
        return {"name": "Unknown", "diagnosis": "Not Found", "grade": "N/A", "iep_date": "N/A"}

def generate_graph_insights(df):
    """Sends the raw data to AI to find patterns"""
    data_summary = df.to_string()
    prompt = f"""
    You are a Behavioral Data Scientist. Analyze this student data:
    {data_summary}
    
    Give me 3 short, bulleted insights about trends, triggers, or progress.
    Be professional and direct.
    """
    return llm.invoke(prompt).content

# --- 3. AGENT GRAPH ---
class AgentState(TypedDict):
    user_request: str
    router_decision: str
    final_response: str
    active_agent: str

def router_node(state):
    prompt = f"""
    Analyze request: "{state['user_request']}"
    Classify into: 'compliance', 'history', 'strategy', 'analytics'. 
    Return ONLY the word (lowercase).
    """
    try: decision = llm.invoke(prompt).content.strip().lower()
    except: decision = "strategy"
    
    if "compliance" in decision: return {"router_decision": "compliance"}
    if "history" in decision: return {"router_decision": "history"}
    if "analytics" in decision: return {"router_decision": "analytics"}
    return {"router_decision": "strategy"}

def compliance_agent(state):
    # RESTORED: Professional Compliance Officer Persona
    prompt = f"""
    You are a Special Education Compliance Officer. 
    Check this request against the IDEA Act and US Education Law.
    Request: '{state['user_request']}'
    """
    return {"final_response": llm.invoke(prompt).content, "active_agent": "Compliance Agent"}

def history_agent(state):
    # RESTORED: Clinical Analyst Persona
    pdf = st.session_state.get("pdf_context", "")
    if not pdf: 
        return {"final_response": "📂 Please upload a PDF in the Sidebar first so I can analyze the student's file.", "active_agent": "System"}
    
    prompt = f"""
    You are a Clinical Analyst. 
    SOURCE DOCUMENT (Student Record):
    {pdf[:20000]} 
    
    User Question: '{state['user_request']}'
    Answer using ONLY the source document above.
    """
    return {"final_response": llm.invoke(prompt).content, "active_agent": "History Agent"}

def strategy_agent(state):
    # RESTORED: Empathetic Teacher Persona
    prompt = f"""
    You are an Empathetic Special Education Teacher. 
    Create a practical teaching strategy, behavior plan, or email draft for: '{state['user_request']}'
    """
    return {"final_response": llm.invoke(prompt).content, "active_agent": "Strategy Agent"}

# workflow setup
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("compliance", compliance_agent)
workflow.add_node("history", history_agent)
workflow.add_node("strategy", strategy_agent)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x['router_decision'], 
    {"compliance": "compliance", "history": "history", "strategy": "strategy", "analytics": "strategy"}) # fallback analytics to strategy for chat
workflow.add_edge("compliance", END)
workflow.add_edge("history", END)
workflow.add_edge("strategy", END)
app_graph = workflow.compile()

# --- 4. UI ---
st.title("🧠 NeuroSync OS")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System Ready. I can help with Compliance, History Analysis, or Teaching Strategies.", "agent": "System"}]
if "profile" not in st.session_state:
    st.session_state.profile = {"name": "Waiting...", "diagnosis": "---", "grade": "---", "iep_date": "---"}

# SIDEBAR
with st.sidebar:
    st.title("📂 Case File")
    uploaded_file = st.file_uploader("Upload IEP (PDF)", type="pdf")
    if uploaded_file:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join([page.extract_text() for page in reader.pages])
            st.session_state["pdf_context"] = text
            if st.session_state.profile["name"] == "Waiting...":
                with st.spinner("Analyzing..."):
                    st.session_state.profile = extract_profile_from_pdf(text)
            st.toast("PDF Loaded!", icon="✅")
        except: st.error("PDF Error")

    with st.container(border=True):
        st.subheader(st.session_state.profile["name"])
        st.caption(f"Diagnosis: {st.session_state.profile['diagnosis']}")
        c1, c2 = st.columns(2)
        c1.metric("Grade", st.session_state.profile["grade"])
        c2.metric("IEP Due", st.session_state.profile["iep_date"])

    st.divider()
    st.write("🎙️ **Voice Command**")
    audio_text = speech_to_text(language='en', just_once=True, key='mic_sidebar')
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# TABS
tab1, tab2 = st.tabs(["💬 Assistant", "📊 Live Analytics"])

with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "agent" in msg and msg["agent"] != "User": st.caption(f"⚡ {msg['agent']}")
            st.write(msg["content"])

    final_input = audio_text if audio_text else st.chat_input("Type request...")
    
    if final_input:
        st.session_state.messages.append({"role": "user", "content": final_input, "agent": "User"})
        with st.chat_message("user"): st.write(final_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = app_graph.invoke({"user_request": final_input})
                st.caption(f"⚡ {res['active_agent']}")
                st.write(res['final_response'])
                st.session_state.messages.append({"role": "assistant", "content": res['final_response'], "agent": res['active_agent']})

# --- FEATURE 2 & 3: REAL DATA + AI INSIGHTS ---
with tab2:
    st.header("Behavioral Trends")
    
    # 1. Data Source (Upload or Mock)
    data_file = st.file_uploader("Upload Data Log (.csv)", type="csv")
    
    if data_file:
        # REAL DATA MODE
        df = pd.read_csv(data_file)
        st.success("✅ Analyzing Uploaded Data")
    else:
        # DEMO MODE (Default)
        st.info("ℹ️ Using Demo Data. Upload a CSV to see real analytics.")
        data = {
            "Day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
            "Incidents": [3, 1, 5, 2, 4],
            "Focus_Score": [40, 70, 30, 80, 50]
        }
        df = pd.DataFrame(data)

    # 2. Visualize
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Incident Frequency")
        st.bar_chart(df.set_index(df.columns[0])[df.columns[1]], color="#ff4b4b")
    with c2:
        st.subheader("Focus Trends")
        st.line_chart(df.set_index(df.columns[0])[df.columns[2]], color="#4F46E5")

    # 3. AI INSIGHTS (The New Feature)
    st.subheader("🤖 AI Data Analysis")
    if st.button("Generate Insights"):
        with st.spinner("Consulting Data Scientist Agent..."):
            insight = generate_graph_insights(df)
            st.markdown(insight)