"""
AutoStream Agent — CLI Entrypoint
Run with: python main.py [--provider anthropic|openai|google] [--model MODEL_NAME]
"""

import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="AutoStream Conversational AI Agent")
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "google"],
        help="LLM provider to use (default: anthropic)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name (optional — uses sensible default per provider)"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the welcome banner"
    )
    return parser.parse_args()


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          AutoStream AI Sales Assistant — Powered by Inflx    ║
║          Type 'quit' or 'exit' to end the session            ║
║          Type 'reset' to start a new conversation            ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def validate_env(provider: str):
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    key = env_map[provider]
    if not os.environ.get(key):
        print(f"\n[ERROR] Missing API key: {key}")
        print(f"Set it in your .env file or export it as an environment variable.\n")
        sys.exit(1)


def run_cli():
    args = parse_args()
    validate_env(args.provider)

    if not args.no_banner:
        print_banner()

    print(f"[CONFIG] Provider: {args.provider} | Model: {args.model or 'default'}")
    print(f"[CONFIG] Knowledge base: ./knowledge_base/autostream_kb.json\n")

    # Import here so env is loaded first
    from agent.graph import AutoStreamAgent

    agent = AutoStreamAgent(provider=args.provider, model=args.model)

    print("Aria: Hello! Welcome to AutoStream. I'm Aria, your AI assistant. How can I help you today?\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAria: Thanks for chatting! Have a great day. 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("\nAria: Thanks for your interest in AutoStream! Feel free to reach out anytime. Goodbye! 👋")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("\n[Session reset. Starting fresh conversation.]\n")
            print("Aria: Hello again! How can I help you with AutoStream today?\n")
            continue

        if user_input.lower() == "debug":
            print(f"\n[DEBUG] Turn: {agent.turn_count}")
            print(f"[DEBUG] Intent: {agent._state.get('current_intent')}")
            print(f"[DEBUG] Lead active: {agent._state.get('lead_collection_active')}")
            print(f"[DEBUG] Lead captured: {agent._state.get('lead_captured')}")
            print(f"[DEBUG] Collected: {agent._state.get('lead_collector_state', {}).get('collected', {})}")
            print()
            continue

        try:
            response = agent.chat(user_input)
            print(f"\nAria: {response}\n")

            if agent.is_lead_captured:
                print("\n[✓ Lead successfully captured. Session will continue for any follow-up questions.]\n")

        except Exception as e:
            print(f"\n[ERROR] Agent encountered an issue: {e}")
            print("Please try again or type 'reset' to restart.\n")


if __name__ == "__main__":
    run_cli()
