"""
AutoStream Agent — LangGraph Graph Builder
Wires all nodes into a conditional state machine graph.
"""

import os
from functools import partial
from typing import Literal

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.intent_classifier import Intent
from agent.nodes import (
    node_classify_intent,
    node_retrieve_context,
    node_handle_lead_collection,
    node_generate_response,
    node_activate_lead_flow,
)


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------

def build_llm(provider: str = "anthropic", model: str = None):
    """
    Builds and returns an LLM client.
    Supports: anthropic (Claude), openai (GPT), google (Gemini).
    Set the corresponding API key in environment variables.
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-haiku-4-5",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            temperature=0.4,
            max_tokens=1024,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.4,
            max_tokens=1024,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-flash",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0.4,
            max_output_tokens=1024,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'anthropic', 'openai', or 'google'.")


# ---------------------------------------------------------------------------
# Conditional Edge Routing
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> Literal["activate_lead", "retrieve_context", "generate_response"]:
    """Routes to the correct node after intent classification."""
    intent = state.get("current_intent")
    lead_active = state.get("lead_collection_active", False)
    lead_captured = state.get("lead_captured", False)

    # Already in lead collection and not yet done
    if lead_active and not lead_captured:
        return "handle_lead_collection"

    if intent == Intent.HIGH_INTENT_LEAD.value:
        return "activate_lead"

    if intent == Intent.PRODUCT_INQUIRY.value:
        return "retrieve_context"

    return "generate_response"  # GREETING or fallback


def route_after_retrieve(state: AgentState) -> Literal["generate_response"]:
    return "generate_response"


def route_after_activate(state: AgentState) -> Literal["handle_lead_collection"]:
    return "handle_lead_collection"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_agent_graph(llm) -> StateGraph:
    """
    Constructs and compiles the LangGraph state machine.
    """
    builder = StateGraph(AgentState)

    # --- Bind LLM to nodes that need it ---
    classify_node = partial(node_classify_intent, llm=llm)
    lead_node = partial(node_handle_lead_collection, llm=llm)
    response_node = partial(node_generate_response, llm=llm)

    # --- Register nodes ---
    builder.add_node("classify_intent", classify_node)
    builder.add_node("retrieve_context", node_retrieve_context)
    builder.add_node("activate_lead", node_activate_lead_flow)
    builder.add_node("handle_lead_collection", lead_node)
    builder.add_node("generate_response", response_node)

    # --- Entry point ---
    builder.set_entry_point("classify_intent")

    # --- Edges ---
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "activate_lead": "activate_lead",
            "retrieve_context": "retrieve_context",
            "generate_response": "generate_response",
            "handle_lead_collection": "handle_lead_collection",
        }
    )
    builder.add_edge("retrieve_context", "generate_response")
    builder.add_edge("activate_lead", "handle_lead_collection")
    builder.add_edge("handle_lead_collection", END)
    builder.add_edge("generate_response", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# AutoStream Agent Wrapper
# ---------------------------------------------------------------------------

class AutoStreamAgent:
    """
    High-level agent wrapper for CLI and API usage.
    Manages LangGraph graph + persistent state across turns.
    """

    def __init__(self, provider: str = "anthropic", model: str = None):
        self.llm = build_llm(provider=provider, model=model)
        self.graph = build_agent_graph(self.llm)
        self._state: AgentState = {
            "messages": [],
            "current_intent": None,
            "lead_collection_active": False,
            "lead_collector_state": None,
            "lead_captured": False,
            "turn_count": 0,
            "rag_context": None,
            "response": None,
        }

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.
        State is persisted across calls.
        """
        from langchain_core.messages import HumanMessage

        # Append user message to state
        self._state["messages"] = list(self._state.get("messages", [])) + [
            HumanMessage(content=user_message)
        ]

        # Run the graph
        result = self.graph.invoke(self._state)

        # Persist updated state
        self._state = result

        return result.get("response", "I'm sorry, I didn't understand that. Could you rephrase?")

    def reset(self):
        """Resets conversation state for a new session."""
        self._state = {
            "messages": [],
            "current_intent": None,
            "lead_collection_active": False,
            "lead_collector_state": None,
            "lead_captured": False,
            "turn_count": 0,
            "rag_context": None,
            "response": None,
        }

    @property
    def is_lead_captured(self) -> bool:
        return self._state.get("lead_captured", False)

    @property
    def turn_count(self) -> int:
        return self._state.get("turn_count", 0)
