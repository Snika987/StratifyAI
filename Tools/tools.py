from typing import Literal, List, Annotated
from uuid import uuid4
from langchain.tools import tool
from Database.session import get_connection
from langchain_core.tools import InjectedToolArg
from langgraph.config import get_config

ALLOWED_DEPARTMENTS = ["HR", "Finance", "Travel", "IT"]
ALLOWED_PRIORITIES = ["low", "medium", "high", "urgent"]
ALLOWED_STATUS = ["open", "in_progress", "closed"]


@tool
def create_ticket(
    department: Literal["HR", "Finance", "Travel", "IT"],
    description: str,
    priority: Literal["low", "medium", "high", "urgent"] = "medium",
) -> dict:

    """Create a new support ticket for a user in the specified department with a given priority and description."""

    config = get_config()
    user_id = config.get("configurable", {}).get("user_id")

    if not user_id:
        raise ValueError("user_id missing from runtime config")

    if department not in ALLOWED_DEPARTMENTS:
        raise ValueError("Invalid department")

    if priority not in ALLOWED_PRIORITIES:
        raise ValueError("Invalid priority")

    conn = get_connection()
    cur = conn.cursor()

    ticket_id = str(uuid4())

    cur.execute("""
        SELECT id FROM departments WHERE name = %s
    """, (department,))
    dept = cur.fetchone()

    if not dept:
        raise ValueError("Department not found")

    department_id = dept[0]

    cur.execute("""
        INSERT INTO tickets (id, user_id, department_id, description, priority, status)
        VALUES (%s, %s, %s, %s, %s, 'open')
    """, (ticket_id, user_id, department_id, description, priority))

    conn.commit()
    cur.close()
    conn.close()

    return {
        "ticket_id": ticket_id,
        "status": "open",
        "department": department,
        "priority": priority,
    }


@tool
def update_ticket_fields(ticket_id: str, fields: dict) -> dict:
    """Update one or more fields such as status or priority for an existing ticket."""
    conn = get_connection()
    cur = conn.cursor()

    updates = []
    values = []

    for key, value in fields.items():
        if key == "priority" and value not in ALLOWED_PRIORITIES:
            raise ValueError("Invalid priority")
        if key == "status" and value not in ALLOWED_STATUS:
            raise ValueError("Invalid status")

        updates.append(f"{key} = %s")
        values.append(value)

    if not updates:
        return {"message": "No valid fields to update"}

    values.append(ticket_id)

    query = f"""
        UPDATE tickets
        SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        RETURNING id, status, priority
    """

    cur.execute(query, tuple(values))
    updated = cur.fetchone()

    conn.commit()
    cur.close()
    conn.close()

    if not updated:
        raise ValueError("Ticket not found")

    return {
        "ticket_id": updated[0],
        "status": updated[1],
        "priority": updated[2],
    }


@tool
def add_ticket_message(
    ticket_id: str,
    sender: Literal["user", "agent", "system"],
    content: str,
) -> dict:
    """Add a message to an existing ticket conversation from a user, agent, or system."""
    conn = get_connection()
    cur = conn.cursor()

    message_id = str(uuid4())

    cur.execute("""
        INSERT INTO ticket_messages (id, ticket_id, sender, content)
        VALUES (%s, %s, %s, %s)
    """, (message_id, ticket_id, sender, content))

    conn.commit()
    cur.close()
    conn.close()

    return {
        "message_id": message_id,
        "ticket_id": ticket_id,
        "sender": sender,
    }


@tool
def assign_to_department(
    ticket_id: str,
    department: Literal["HR", "Finance", "Travel", "IT"],
) -> dict:
    """Assign an existing ticket to a specific department."""
    if department not in ALLOWED_DEPARTMENTS:
        raise ValueError("Invalid department")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id FROM departments WHERE name = %s
    """, (department,))
    dept = cur.fetchone()

    if not dept:
        raise ValueError("Department not found")

    department_id = dept[0]

    cur.execute("""
        UPDATE tickets
        SET department_id = %s, updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        RETURNING id
    """, (department_id, ticket_id))

    updated = cur.fetchone()

    conn.commit()
    cur.close()
    conn.close()

    if not updated:
        raise ValueError("Ticket not found")

    return {
        "ticket_id": ticket_id,
        "assigned_department": department,
    }


@tool
def close_ticket(ticket_id: str) -> dict:
    """Mark an existing ticket as closed."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE tickets
        SET status = 'closed', updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        RETURNING id
    """, (ticket_id,))

    updated = cur.fetchone()

    conn.commit()
    cur.close()
    conn.close()

    if not updated:
        raise ValueError("Ticket not found")

    return {
        "ticket_id": ticket_id,
        "status": "closed",
    }


@tool
def get_ticket(ticket_id: str) -> dict:
    """Retrieve full details of a ticket including description, status, priority, department, and all messages."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT t.id, t.description, t.status, t.priority, d.name
        FROM tickets t
        JOIN departments d ON t.department_id = d.id
        WHERE t.id = %s
    """, (ticket_id,))
    ticket = cur.fetchone()

    if not ticket:
        raise ValueError("Ticket not found")

    cur.execute("""
        SELECT sender, content, created_at
        FROM ticket_messages
        WHERE ticket_id = %s
        ORDER BY created_at ASC
    """, (ticket_id,))
    messages = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "ticket_id": ticket[0],
        "description": ticket[1],
        "status": ticket[2],
        "priority": ticket[3],
        "department": ticket[4],
        "messages": messages,
    }


@tool
def get_user_ticket_history(user_id: str, limit: int = 3) -> List[dict]:
    """Retrieve the most recent support tickets created by a specific user."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, description, status, priority
        FROM tickets
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT %s
    """, (user_id, limit))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "ticket_id": row[0],
            "description": row[1],
            "status": row[2],
            "priority": row[3],
        }
        for row in rows
    ]

@tool
def search_company_policies(query: str) -> str:
    """
    Search internal company policy documents.
    Use this when user asks about HR, Finance, IT, or Travel policies.
    """
    from RAG.retrieve import build_context

    context = build_context(query)

    if not context:
        return "NOT_FOUND"

    return context