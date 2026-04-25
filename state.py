"""
LangGraph State Schema for AutoStream Agent.
Defines the full state object that persists across conversation turns.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Core state container passed between LangGraph nodes.
    All fields are preserved across turns via LangGraph's state machine.
    """

    # Full conversation history (managed by LangGraph add_messages reducer)
    messages: Annotated[list, add_messages]

    # Detected intent for the current user message
    current_intent: Optional[str]

    # Whether the agent is in lead-collection mode
    lead_collection_active: bool

    # Serialized LeadCollector state (dict representation)
    lead_collector_state: Optional[dict]

    # Whether the lead has been fully captured
    lead_captured: bool

    # Number of conversation turns
    turn_count: int

    # RAG context retrieved for the current query
    rag_context: Optional[str]

    # Final response to send back to user
    response: Optional[str]
