import streamlit as st
import os
from dotenv import load_dotenv
from typing import TypedDict
import PyPDF2
from streamlit_mic_recorder import speech_to_text
import pandas as pd
import json
from supabase import create_client, Client
from fpdf import FPDF
from gtts import gTTS 
import io

# AI Libraries
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroSync OS", page_icon="🧠", layout="wide")
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; color: #4F46E5; }
    .stChatInput { padding-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()

# --- 2. CONNECTIONS ---
try:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url: st.stop()
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # AUTH FIX: Restore Session if available
    if "access_token" in st.session_state:
        try:
            supabase.auth.set_session(st.session_state.access_token, st.session_state.refresh_token)
        except:
            pass # Session expired
except: st.stop()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key: st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_key)

# --- 3. DATABASE FUNCTIONS ---
def login_user(email, password):
    try: 
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        # SAVE SESSION
        st.session_state.access_token = res.session.access_token
        st.session_state.refresh_token = res.session.refresh_token
        return res.user
    except: return None

def signup_user(email, password):
    try: return supabase.auth.sign_up({"email": email, "password": password}).user
    except: return None

def save_message(user_email, role, content, agent_name, student_name):
    supabase.table("chat_logs").insert({
        "user_email": user_email, "role": role, "content": content, 
        "agent_name": agent_name, "student_name": student_name
    }).execute()

def load_history(user_email, student_name):
    res = supabase.table("chat_logs").select("*").eq("user_email", user_email).eq("student_name", student_name).order("created_at", desc=False).execute()
    return [{"role": m['role'], "content": m['content'], "agent_name": m['agent_name']} for m in res.data]

def add_student(user_id, name, diagnosis, grade):
    supabase.table("students").insert({
        "user_id": user_id, "name": name, "diagnosis": diagnosis, "grade": grade
    }).execute()

def get_students(user_id):
    res = supabase.table("students").select("name").eq("user_id", user_id).execute()
    return [s['name'] for s in res.data] if res.data else []

def save_document_text(student_name, text):
    supabase.table("documents").delete().eq("filename", student_name).execute() 
    supabase.table("documents").insert({"filename": student_name, "content_text": text}).execute()

def load_document_text(student_name):
    res = supabase.table("documents").select("content_text").eq("filename", student_name).execute()
    if res.data: return res.data[0]['content_text']
    return None

def create_pdf_report(content, student_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(40, 10, f"NeuroSync Report: {student_name}")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    clean_content = content.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_content)
    return pdf.output()

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    return audio_fp

# --- 4. AGENT LOGIC ---
def extract_profile_from_pdf(text):
    prompt = f"""Extract student details. Return JSON ONLY: {{ "name": "Name", "diagnosis": "Diagnosis", "grade": "Grade", "iep_date": "Date" }} TEXT: {text[:3000]}"""
    try: return json.loads(llm.invoke(prompt).content.replace("```json", "").replace("```", "").strip())
    except: return {"name": "Unknown", "diagnosis": "Not Found", "grade": "-", "iep_date": "-"}

def generate_graph_insights(df):
    return llm.invoke(f"Analyze this data and give 3 insights:\n{df.to_string()}").content

class AgentState(TypedDict):
    user_request: str
    router_decision: str
    final_response: str
    active_agent: str

def router_node(state):
    try: decision = llm.invoke(f"Classify '{state['user_request']}' into: 'compliance', 'history', 'strategy'. Return 1 word.").content.strip().lower()
    except: decision = "strategy"
    if "compliance" in decision: return {"router_decision": "compliance"}
    if "history" in decision: return {"router_decision": "history"}
    return {"router_decision": "strategy"}

def compliance_agent(state):
    return {"final_response": llm.invoke(f"You are a Compliance Officer. Check IDEA Act compliance: {state['user_request']}").content, "active_agent": "Compliance Agent"}

def history_agent(state):
    pdf = st.session_state.get("pdf_context", "")
    if not pdf: return {"final_response": "📂 No document found for this student. Please upload a PDF.", "active_agent": "System"}
    return {"final_response": llm.invoke(f"Context: {pdf[:15000]}. Answer: {state['user_request']}").content, "active_agent": "History Agent"}

def strategy_agent(state):
    return {"final_response": llm.invoke(f"You are a Teacher. Create a strategy: {state['user_request']}").content, "active_agent": "Strategy Agent"}

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("compliance", compliance_agent)
workflow.add_node("history", history_agent)
workflow.add_node("strategy", strategy_agent)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x['router_decision'], {"compliance": "compliance", "history": "history", "strategy": "strategy"})
workflow.add_edge("compliance", END)
workflow.add_edge("history", END)
workflow.add_edge("strategy", END)
app_graph = workflow.compile()

# --- 5. UI ---
if "user" not in st.session_state: st.session_state.user = None

if not st.session_state.user:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.title("🧠 NeuroSync Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        b1, b2 = st.columns(2)
        if b1.button("Login", type="primary"):
            user = login_user(email, password)
            if user: st.session_state.user = user; st.rerun()
            else: st.error("Failed")
        if b2.button("Sign Up"):
            user = signup_user(email, password)
            if user: st.success("Created!")
            else: st.error("Error")
else:
    with st.sidebar:
        st.title("NeuroSync OS")
        st.divider()
        st.header("Student Selection")
        
        my_students = get_students(st.session_state.user.id)
        if not my_students:
            st.warning("No students found.")
            new_name = st.text_input("Add Student Name")
            new_diag = st.text_input("Diagnosis")
            if st.button("Add Student"):
                add_student(st.session_state.user.id, new_name, new_diag, "5th")
                st.rerun()
            selected_student = None
        else:
            options = my_students + ["+ Add New Student"]
            selected_student = st.selectbox("Active Student", options)
            
            if selected_student == "+ Add New Student":
                new_name = st.text_input("Student Name")
                new_diag = st.text_input("Diagnosis")
                if st.button("Save Student"):
                    add_student(st.session_state.user.id, new_name, new_diag, "5th")
                    st.rerun()
            else:
                if "current_student" not in st.session_state or st.session_state.current_student != selected_student:
                    st.session_state.current_student = selected_student
                    st.session_state.messages = load_history(st.session_state.user.email, selected_student)
                    saved_pdf = load_document_text(selected_student)
                    if saved_pdf: st.session_state["pdf_context"] = saved_pdf
                    else: st.session_state["pdf_context"] = ""

                uploaded_file = st.file_uploader("Update Record (PDF)", type="pdf")
                if uploaded_file:
                    reader = PyPDF2.PdfReader(uploaded_file)
                    text = "".join([page.extract_text() for page in reader.pages])
                    st.session_state["pdf_context"] = text
                    save_document_text(selected_student, text)
                    st.session_state.profile = extract_profile_from_pdf(text)
                    st.toast("Record Saved!", icon="☁️")

        if st.button("Logout"): st.session_state.user = None; st.rerun()

    if selected_student and selected_student != "+ Add New Student":
        st.title(f"Case: {selected_student}")
        tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Analytics"])

        with tab1:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg.get("agent_name"): st.caption(f"⚡ {msg['agent_name']}")
                    st.write(msg["content"])
                    if msg["role"] == "assistant":
                        pdf_bytes = create_pdf_report(msg["content"], selected_student)
                        c_dl, c_play = st.columns([1, 4])
                        c_dl.download_button(label="📄 PDF", data=bytes(pdf_bytes), file_name="report.pdf", mime="application/pdf", key=f"dl_{str(msg)}")
            
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                last_msg = st.session_state.messages[-1]["content"]
                if st.button("🔊 Listen"):
                    audio_fp = text_to_speech(last_msg[:500]) 
                    st.audio(audio_fp, format='audio/mp3')

            audio_text = speech_to_text(language='en', just_once=True, key='mic')
            prompt = st.chat_input("Type request...") or audio_text

            if prompt:
                st.chat_message("user").write(prompt)
                save_message(st.session_state.user.email, "user", prompt, None, selected_student)
                st.session_state.messages.append({"role": "user", "content": prompt, "agent_name": "User"})

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            result = app_graph.invoke({"user_request": prompt})
                            response = result['final_response']
                            agent = result['active_agent']
                            st.caption(f"⚡ {agent}")
                            st.write(response)
                            save_message(st.session_state.user.email, "assistant", response, agent, selected_student)
                            st.session_state.messages.append({"role": "assistant", "content": response, "agent_name": agent})
                            st.rerun()
                        except Exception as e:
                            st.error(f"AI Error: {e}")

        with tab2:
            st.header("Behavioral Trends")
            data_file = st.file_uploader("Upload Data Log (.csv)", type="csv")
            if data_file:
                df = pd.read_csv(data_file)
                st.bar_chart(df.set_index(df.columns[0])[df.columns[1]])
                if st.button("Analyze Data"):
                    st.write(generate_graph_insights(df))
            else:
                st.info("Upload a CSV to view analytics.")