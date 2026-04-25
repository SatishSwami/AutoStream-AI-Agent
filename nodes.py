"""
LangGraph Node Implementations for AutoStream Agent.
Each function is a pure node: takes AgentState, returns partial state updates.
"""

import json
import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.state import AgentState
from agent.intent_classifier import Intent, classify_intent
from agent.rag_pipeline import get_rag_pipeline
from tools.lead_capture import LeadCollector


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """You are Aria, the AI sales assistant for AutoStream — a SaaS platform that provides automated video editing tools for content creators.

Your personality: Friendly, concise, helpful, and professional. You speak like a knowledgeable human, not a robot.

Your primary goals:
1. Help users understand AutoStream's features and pricing
2. Answer questions accurately using only the provided knowledge base context
3. Identify when users show genuine interest in signing up
4. Collect their contact details naturally and warmly when they're ready

STRICT RULES:
- NEVER make up features, prices, or policies not found in the context
- NEVER call the lead capture tool unless ALL THREE fields (name, email, platform) are collected
- NEVER ask for multiple fields at once — ask one at a time
- If unsure, say you'll escalate to the team rather than guessing
- Keep responses concise (2-4 sentences for simple queries, slightly more for detailed ones)
"""

SYSTEM_PROMPT_GREETING = SYSTEM_PROMPT_BASE + """
The user has just greeted you. Respond warmly, briefly introduce AutoStream, and invite them to ask questions.
"""

SYSTEM_PROMPT_PRODUCT = SYSTEM_PROMPT_BASE + """
Use the following context from the AutoStream knowledge base to answer the user's question:

{context}

Answer directly and accurately. If the context doesn't cover the question, say so honestly.
"""

SYSTEM_PROMPT_LEAD = SYSTEM_PROMPT_BASE + """
The user has shown strong interest in signing up. You are now in lead qualification mode.

Lead collection status:
{lead_status}

Next field to collect: {next_field}

{next_prompt}

Continue the conversation naturally. Acknowledge what they said, then ask for the next missing field.
Do NOT ask for fields already collected. Do NOT call any tool yet.
"""

SYSTEM_PROMPT_CAPTURE = SYSTEM_PROMPT_BASE + """
You have successfully collected all lead information:
{lead_data}

The lead has been captured in our CRM system (Lead ID: {lead_id}).

Write a warm, enthusiastic closing message:
1. Confirm their details (name, email, platform)
2. Tell them what happens next (team will reach out within 24 hours)
3. Encourage them to explore the Pro trial in the meantime
"""


# ---------------------------------------------------------------------------
# Helper: Extract field from user message
# ---------------------------------------------------------------------------

def _extract_field_from_message(message: str, field: str, collector: LeadCollector) -> Optional[str]:
    """
    Tries to extract a specific field value from the user's message.
    Returns extracted value or None.
    """
    message = message.strip()

    if field == "email":
        match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", message)
        return match.group(0) if match else None

    if field == "platform":
        platforms = ["youtube", "instagram", "tiktok", "twitter", "facebook",
                     "linkedin", "twitch", "podcast", "vimeo", "snapchat"]
        for p in platforms:
            if p in message.lower():
                return p.capitalize()
        # If the message is short, treat the whole thing as a platform answer
        if len(message.split()) <= 5:
            return message

    if field == "name":
        # If the message is a short phrase (1-4 words), treat it as a name
        words = message.split()
        if 1 <= len(words) <= 4 and all(w[0].isupper() or w[0].isalpha() for w in words if w):
            return message
        # Check for patterns like "I'm [Name]" or "my name is [Name]"
        patterns = [
            r"(?:i'm|i am|my name is|call me|it's|its)\s+([A-Za-z]+(?: [A-Za-z]+)?)",
        ]
        for p in patterns:
            match = re.search(p, message, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()

    return None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_classify_intent(state: AgentState, llm) -> dict:
    """
    Node 1: Classify user intent from the latest message.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"current_intent": Intent.GREETING.value}

    latest_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_human = msg.content
            break

    if not latest_human:
        return {"current_intent": Intent.GREETING.value}

    conversation_history = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in messages[:-1]
    ]

    # If already in lead collection, maintain that intent unless user explicitly exits
    if state.get("lead_collection_active") and not state.get("lead_captured"):
        intent = Intent.HIGH_INTENT_LEAD
    else:
        intent = classify_intent(
            user_message=latest_human,
            conversation_history=conversation_history,
            llm_client=llm,
            use_llm=True,
        )

    return {
        "current_intent": intent.value,
        "turn_count": state.get("turn_count", 0) + 1,
    }


def node_retrieve_context(state: AgentState) -> dict:
    """
    Node 2: Retrieve relevant RAG context if intent requires knowledge base lookup.
    """
    intent = state.get("current_intent")
    messages = state.get("messages", [])

    if intent == Intent.GREETING.value:
        return {"rag_context": None}

    latest_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_human = msg.content
            break

    if not latest_human:
        return {"rag_context": None}

    rag = get_rag_pipeline()
    context = rag.get_context_string(latest_human, top_k=3)
    return {"rag_context": context}


def node_handle_lead_collection(state: AgentState, llm) -> dict:
    """
    Node 3: Handle progressive lead field collection.
    Extracts fields from user messages, prompts for missing ones,
    and fires the capture tool once all fields are collected.
    """
    messages = state.get("messages", [])
    collector_data = state.get("lead_collector_state")
    collector = LeadCollector.from_dict(collector_data) if collector_data else LeadCollector()

    # Get latest human message
    latest_human = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_human = msg.content
            break

    # Try to extract missing fields from current message
    for field in collector.missing_fields():
        value = _extract_field_from_message(latest_human, field, collector)
        if value:
            collector.set_field(field, value)
            break  # Only extract one field per turn

    # All fields collected — execute capture
    if collector.is_complete() and not collector.is_captured:
        result = collector.execute_capture()

        # Generate closing message
        system = SYSTEM_PROMPT_CAPTURE.format(
            lead_data=json.dumps(collector.collected, indent=2),
            lead_id=result.get("lead_id", "N/A"),
        )
        response = llm.invoke([SystemMessage(content=system)] + messages)
        closing_message = response.content

        return {
            "lead_collector_state": collector.to_dict(),
            "lead_captured": True,
            "lead_collection_active": False,
            "messages": [AIMessage(content=closing_message)],
            "response": closing_message,
        }

    # Still collecting — prompt for next field
    missing = collector.missing_fields()
    lead_status_lines = []
    for f in ["name", "email", "platform"]:
        status = collector.collected.get(f, "NOT YET COLLECTED")
        lead_status_lines.append(f"  - {f}: {status}")
    lead_status = "\n".join(lead_status_lines)

    system = SYSTEM_PROMPT_LEAD.format(
        lead_status=lead_status,
        next_field=missing[0] if missing else "none",
        next_prompt=collector.next_prompt(),
    )

    # Include RAG context if available
    rag_context = state.get("rag_context")
    if rag_context and "No relevant" not in rag_context:
        system += f"\n\nAdditional context if relevant:\n{rag_context}"

    response = llm.invoke([SystemMessage(content=system)] + messages)
    reply = response.content

    return {
        "lead_collector_state": collector.to_dict(),
        "lead_collection_active": True,
        "messages": [AIMessage(content=reply)],
        "response": reply,
    }


def node_generate_response(state: AgentState, llm) -> dict:
    """
    Node 4: Generate a response for GREETING or PRODUCT_INQUIRY intents.
    """
    intent = state.get("current_intent")
    messages = state.get("messages", [])
    rag_context = state.get("rag_context")

    if intent == Intent.GREETING.value:
        system = SYSTEM_PROMPT_GREETING
    else:
        context = rag_context or "No specific context available. Provide a general, helpful response."
        system = SYSTEM_PROMPT_PRODUCT.format(context=context)

    response = llm.invoke([SystemMessage(content=system)] + messages)
    reply = response.content

    return {
        "messages": [AIMessage(content=reply)],
        "response": reply,
    }


def node_activate_lead_flow(state: AgentState) -> dict:
    """
    Node 5: Transition into lead collection mode when high intent is detected.
    Initializes a fresh LeadCollector if one doesn't already exist.
    """
    existing = state.get("lead_collector_state")
    if not existing or not existing.get("collected"):
        fresh_collector = LeadCollector()
        return {
            "lead_collection_active": True,
            "lead_collector_state": fresh_collector.to_dict(),
        }
    return {"lead_collection_active": True}
