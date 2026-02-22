# app.py
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import streamlit as st

# Backend function already available in your project
from Agents.graph import invoke_agent


def _init_session() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"thread_{uuid.uuid4()}"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _new_conversation() -> None:
    st.session_state.thread_id = f"thread_{uuid.uuid4()}"
    st.session_state.chat_history = []


def _to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_messages(state: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for msg in state.get("messages", []) or []:
        role = "assistant"
        content = ""

        if isinstance(msg, dict):
            raw_role = str(msg.get("role", "")).lower()
            if raw_role in {"human", "user"}:
                role = "user"
            elif raw_role in {"ai", "assistant"}:
                role = "assistant"
            else:
                role = "assistant"
            content = _to_text(msg.get("content", ""))

        else:
            msg_type = str(getattr(msg, "type", "")).lower()
            if msg_type in {"human", "user"}:
                role = "user"
            elif msg_type in {"ai", "assistant"}:
                role = "assistant"
            else:
                role = "assistant"
            content = _to_text(getattr(msg, "content", ""))

        if content.strip():
            out.append({"role": role, "content": content.strip()})
    return out


def _extract_assistant_reply(state: Dict[str, Any]) -> str:
    parsed = _extract_messages(state)
    for m in reversed(parsed):
        if m["role"] == "assistant":
            return m["content"]
    return "Done."


def _render_assistant_extras(entry: Dict[str, Any]) -> None:
    ticket_id = entry.get("ticket_id")
    ticket_status = entry.get("status")
    rag_context = entry.get("rag_context")

    if ticket_id:
        st.markdown(
            f"""
            <div class="ticket-box">
              <div class="ticket-title">Ticket Created</div>
              <div><b>Ticket ID:</b> {ticket_id}</div>
              <div><b>Status:</b> {ticket_status or "open"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if rag_context:
        st.markdown(
            f"""
            <div class="policy-box">
              <div class="policy-title">Policy Context</div>
              <div class="policy-text">{rag_context}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.set_page_config(page_title="Enterprise Support Agent", page_icon="🛠", layout="wide")
_init_session()

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #0b1020 0%, #0f172a 100%); color: #e5e7eb; }
      .header-wrap { padding: 0.25rem 0 0.5rem 0; }
      .header-title { font-size: 1.4rem; font-weight: 700; color: #f8fafc; }
      .header-sub { color: #94a3b8; font-size: 0.9rem; margin-top: 0.2rem; }
      .ticket-box {
        border: 1px solid #14532d; background: #052e16; border-radius: 10px;
        padding: 12px; margin-top: 8px; margin-bottom: 8px; color: #dcfce7;
      }
      .ticket-title { font-weight: 700; margin-bottom: 6px; color: #86efac; }
      .policy-box {
        border: 1px solid #1e3a8a; background: #0c1a4b; border-radius: 10px;
        padding: 12px; margin-top: 8px; margin-bottom: 8px; color: #dbeafe;
      }
      .policy-title { font-weight: 700; margin-bottom: 6px; color: #93c5fd; }
      .policy-text { white-space: pre-wrap; line-height: 1.4; }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([6, 2])
with col1:
    st.markdown(
        """
        <div class="header-wrap">
          <div class="header-title">Enterprise LangGraph Support Agent</div>
          <div class="header-sub">Policy Q&A, incident triage, and ticket workflows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    if st.button("New Conversation", use_container_width=True):
        _new_conversation()
        st.rerun()

with st.sidebar:
    st.caption("Session")
    st.code(f"user_id: {st.session_state.user_id}\nthread_id: {st.session_state.thread_id}", language="text")

# Render full chat history
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant":
            _render_assistant_extras(entry)

user_prompt = st.chat_input("Ask policy questions, report incidents, or request support...")
if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            state = invoke_agent(
                user_query=user_prompt,
                thread_id=st.session_state.thread_id,
                user_id=st.session_state.user_id,
            )

        assistant_text = _extract_assistant_reply(state)
        st.markdown(assistant_text)

        assistant_entry = {
            "role": "assistant",
            "content": assistant_text,
            "ticket_id": state.get("ticket_id"),
            "status": state.get("status"),
            "rag_context": state.get("rag_context"),
            "confirmed": state.get("confirmed"),
            "tasks": state.get("tasks"),
        }
        _render_assistant_extras(assistant_entry)
        st.session_state.chat_history.append(assistant_entry)
