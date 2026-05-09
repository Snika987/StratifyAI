# Enterprise Ticketing Agent

> An enterprise support agent built using LangGraph and LangChain for handling policy queries, incident reports and service requests through a modular agentic workflow.

---

# ✨ Features

- Multi-intent request handling
- Human-in-the-loop confirmation
- Retrieval-Augmented Generation (RAG)
- Persistent multi-turn conversation state
- Ticket creation workflows
- Observability using Opik
- Semantic caching and tool caching

---

# 🛠️ Tech Stack

- Python
- LangGraph
- LangChain
- PostgreSQL
- NumPy
- Streamlit
- Opik

---
# 🚀 Installation Guide

This guide explains how to set up and run the Enterprise Ticketing Agent locally.

---

# 1️⃣ Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

---

# 2️⃣ Create a Virtual Environment

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4️⃣ Setup PostgreSQL

Ensure PostgreSQL is installed and running.

Create the database:

```sql
CREATE DATABASE Ticketing_Agent;
```

---

# 5️⃣ Configure Environment Variables

Create a `.env` file in the project root directory:

```env
DB_NAME=Ticketing_Agent
DB_USER=your_username
DB_PASSWORD=poassword
DB_HOST=localhost
DB_PORT=5432

MISTRAL_API_KEY=your_mistral_api_key
OPIK_API_KEY=your_opik_api_key
```

---

# 6️⃣ Initialize Database Tables

Run your database initialization or migration script if applicable.

Example:

```bash
python init_db.py
```

---

# 7️⃣ Run the Application

## Streamlit UI

```bash
streamlit run app.py
```

## CLI Testing

```bash
python main.py
```

---

# 📊 Observability

The project integrates Opik for:

- LangGraph execution tracing
- Workflow observability
- Retrieval monitoring
- Tool execution tracking




