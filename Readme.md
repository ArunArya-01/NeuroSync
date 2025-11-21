# 🧠 NeuroSync: The Agentic Special Education OS

**NeuroSync** is a Multi-Agent AI System designed to solve the fragmentation in Special Education management. It uses a **Router-Agent Architecture** to autonomously handle legal compliance, student history analysis, and pedagogical strategy generation.

Built for the **"Unsolved Problems of 2025"**, this system replaces static forms with active, reasoning agents.

---

## ⚡ Features & Architecture

NeuroSync uses **LangGraph** to orchestrate a team of specialized AI agents. It does not use a single generic LLM; instead, it routes tasks to experts.

### The Agent Team
| Badge | Agent Name | Model Engine | Role |
| :--- | :--- | :--- | :--- |
| 🚦 | **The Router** | Llama 3.3 (70B) | The "Traffic Cop". It analyzes user intent (zero-shot) and routes the request to the correct expert. |
| ⚖️ | **Compliance Agent** | Llama 3.3 (Logic Mode) | The "Lawyer". Checks requests against the **IDEA Act** and **Section 504** to prevent lawsuits and ensure rights. |
| 📂 | **History Agent** | Llama 3.3 (128k Context) | The "Analyst". Reads student context (diagnosis, past incidents) to answer questions about background. |
| 🍎 | **Strategy Agent** | Llama 3.3 (Creative Mode) | The "Teacher". Generates empathetic emails, lesson plans, and behavioral accommodations. |

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit (Python-based Reactive UI)
* **Orchestration:** LangGraph (Stateful Multi-Agent Graph)
* **LLM Engine:** Groq Cloud (Running Llama 3.3 70B at ~300 tokens/sec)
* **Framework:** LangChain
* **Language:** Python 3.10+

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/neurosync.git](https://github.com/yourusername/neurosync.git)
cd neurosync