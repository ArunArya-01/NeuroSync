import streamlit as st
import os
from dotenv import load_dotenv
from typing import TypedDict
import PyPDF2 # PDF Reader
from streamlit_mic_recorder import speech_to_text # Voice Input

# AI Libraries
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroSync", page_icon="🧠", layout="wide")

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

st.title("🧠 NeuroSync: Agentic Special Ed OS")

if not groq_key:
    st.error("🚨 CRITICAL ERROR: GROQ_API_KEY is missing.")
    st.stop()

# --- 2. SETUP MODELS ---
try:
    # We use Llama 3.3 for everything (It's smart and free)
    llm = ChatGroq(
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
    Classify into:
    - 'compliance' (laws, rights, IDEA act, suspension)
    - 'history' (records, diagnosis, reports, specific details about student)
    - 'strategy' (teaching, emails, lesson plans, advice)
    Return ONLY the word (lowercase).
    """
    try:
        decision = llm.invoke(prompt).content.strip().lower()
    except:
        decision = "strategy"
    
    if "compliance" in decision: return {"router_decision": "compliance"}
    if "history" in decision: return {"router_decision": "history"}
    return {"router_decision": "strategy"}

def compliance_agent(state):
    prompt = f"You are a Special Ed Lawyer. Check compliance for: '{state['user_request']}'"
    res = llm.invoke(prompt).content
    return {"final_response": res, "active_agent": "Compliance Agent"}

def history_agent(state):
    # REAL RAG: Check memory for PDF text
    pdf_context = st.session_state.get("pdf_context", "")
    
    if not pdf_context:
        return {
            "final_response": "📂 No PDF found. Please upload a report in the sidebar so I can answer this.", 
            "active_agent": "System"
        }
    
    prompt = f"""
    You are a Clinical Analyst. 
    SOURCE DOCUMENT:
    {pdf_context[:20000]} # Limit context size
    
    User Question: '{state['user_request']}'
    Answer using ONLY the source document.
    """
    res = llm.invoke(prompt).content
    return {"final_response": res, "active_agent": "History Agent (RAG)"}

def strategy_agent(state):
    prompt = f"You are an Empathetic Teacher. Create a strategy/email for: '{state['user_request']}'"
    res = llm.invoke(prompt).content
    return {"final_response": res, "active_agent": "Strategy Agent"}

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

# --- 5. UI & SIDEBAR ---
with st.sidebar:
    st.header("📂 Student Case File")
    
    # --- FEATURE 1: REAL PDF READER ---
    uploaded_file = st.file_uploader("Upload IEP/Psych Report", type="pdf")
    
    if uploaded_file:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            st.session_state["pdf_context"] = text
            st.success(f"✅ Processed {len(reader.pages)} pages")
        except Exception as e:
            st.error("Failed to read PDF")
            
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ready. Upload a PDF to analyze history, or ask me for strategies.", "agent": "System"}]

# Display Messages
for index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if "agent" in msg and msg["agent"] != "User":
            st.caption(f"⚡ {msg['agent']}")
        st.write(msg["content"])
        
        # --- FEATURE 3: DOWNLOAD BUTTON ---
        if msg["role"] == "assistant" and msg["agent"] != "System":
            st.download_button(
                label="📄 Download",
                data=msg["content"],
                file_name=f"neurosync_response_{index}.txt",
                mime="text/plain",
                key=f"download_{index}"
            )

# --- FEATURE 2: VOICE INPUT ---
st.write("---")
col1, col2 = st.columns([1, 6])
with col1:
    st.write("🎤 **Speak:**")
    audio_text = speech_to_text(language='en', just_once=True, key='mic_input')

with col2:
    # Logic: Use audio text if available, otherwise use text input
    if audio_text:
        user_input = audio_text
        # Auto-submit logic simulation
        submit = True
    else:
        user_input = st.chat_input("Type or Speak your request...")
        submit = False

# Handle Processing
if user_input:
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": user_input, "agent": "User"})
    with st.chat_message("user"):
        st.write(user_input)

    # 2. Run AI
    with st.chat_message("assistant"):
        with st.spinner("Agents are coordinating..."):
            try:
                result = app_graph.invoke({"user_request": user_input})
                response = result['final_response']
                agent = result['active_agent']
                
                st.caption(f"⚡ {agent}")
                st.write(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response, "agent": agent})
                
                # Show download button for the new message immediately
                st.download_button(
                    label="📄 Download",
                    data=response,
                    file_name="neurosync_response_latest.txt",
                    mime="text/plain"
                )
                
                # Force rerun to update the chat list nicely
                if submit:
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error: {e}")