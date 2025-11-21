import streamlit as st
import os
from dotenv import load_dotenv
from typing import TypedDict

# AI Libraries
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroSync", page_icon="🧠", layout="wide")

# Load keys
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# UI Header
st.title("🧠 NeuroSync OS")

# Check Key
if not groq_key:
    st.error("🚨 CRITICAL ERROR: GROQ_API_KEY is missing.")
    st.stop()

# --- 2. SETUP MODELS (ALL GROQ) ---
try:
    # FAST MODEL (For Routing & Strategy)
    llm_fast = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0, 
        api_key=groq_key
    )
    
    # REASONING MODEL (For Compliance)
    llm_logic = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.1, 
        api_key=groq_key
    )
    
    # CONTEXT MODEL (For History) - We use Llama here too now!
    # Llama 3.3 has 128k context, which is plenty for text history.
    llm_history = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0, 
        api_key=groq_key
    )

except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

# --- 3. AGENT LOGIC ---
class AgentState(TypedDict):
    user_request: str
    router_decision: str
    final_response: str
    active_agent: str

def router_node(state):
    prompt = f"""
    Analyze request: "{state['user_request']}"
    Classify into: 'compliance', 'history', or 'strategy'.
    Return ONLY the word (lowercase).
    """
    try:
        decision = llm_fast.invoke(prompt).content.strip().lower()
    except:
        decision = "strategy"
    
    if "compliance" in decision: return {"router_decision": "compliance"}
    if "history" in decision: return {"router_decision": "history"}
    return {"router_decision": "strategy"}

def compliance_agent(state):
    prompt = f"You are a Special Ed Lawyer. Check compliance for: '{state['user_request']}'"
    res = llm_logic.invoke(prompt).content
    return {"final_response": res, "active_agent": "Compliance Agent (Llama)"}

def history_agent(state):
    # Now uses Groq (Llama) instead of Google
    prompt = f"""
    You are a Clinical Analyst. 
    CONTEXT: Student 'Alex Doe' has ADHD (Combined Type). Diagnosed 2023. Struggles with focus.
    
    User Question: '{state['user_request']}'
    """
    res = llm_history.invoke(prompt).content
    return {"final_response": res, "active_agent": "History Agent (Llama)"}

def strategy_agent(state):
    prompt = f"You are an Empathetic Teacher. Create a strategy for: '{state['user_request']}'"
    res = llm_fast.invoke(prompt).content
    return {"final_response": res, "active_agent": "Strategy Agent (Llama)"}

# --- 4. BUILD GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("compliance", compliance_agent)
workflow.add_node("history", history_agent)
workflow.add_node("strategy", strategy_agent)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x['router_decision'], 
    {"compliance": "compliance", "history": "history", "strategy": "strategy"})
workflow.add_edge("compliance", END)
workflow.add_edge("history", END)
workflow.add_edge("strategy", END)
app_graph = workflow.compile()

# --- 5. UI RENDERING ---
with st.sidebar:
    st.header("System Status")
    st.success("✅ All Agents Online (Groq Engine)")
    if st.button("Clear Chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System Ready.", "agent": "System"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "agent" in msg and msg["agent"] != "User":
            st.caption(f"⚡ {msg['agent']}")
        st.write(msg["content"])

if prompt := st.chat_input("Ask here..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "agent": "User"})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            try:
                result = app_graph.invoke({"user_request": prompt})
                st.caption(f"⚡ {result['active_agent']}")
                st.write(result['final_response'])
                st.session_state.messages.append({"role": "assistant", "content": result['final_response'], "agent": result['active_agent']})
            except Exception as e:
                st.error(f"Error: {e}")