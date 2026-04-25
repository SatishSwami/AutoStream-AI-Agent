"""
Lead Capture Tool for AutoStream Agent.
Simulates a backend CRM API call when a user qualifies as a high-intent lead.
The tool enforces field collection before executing.
"""

import re
import json
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Mock API Function (per assignment spec)
# ---------------------------------------------------------------------------

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates a backend CRM lead capture API call.
    In production, this would POST to a CRM like HubSpot / Salesforce.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    lead_data = {
        "status": "success",
        "lead_id": f"LEAD-{abs(hash(email)) % 100000:05d}",
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": timestamp,
        "assigned_plan": "Pro",
        "source": "AutoStream-Agent-v1",
    }
    print(f"\n{'='*60}")
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    print(f"Lead ID: {lead_data['lead_id']} | Captured at: {timestamp}")
    print(f"{'='*60}\n")
    return lead_data


# ---------------------------------------------------------------------------
# Field Validators
# ---------------------------------------------------------------------------

def validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))


def validate_name(name: str) -> bool:
    return len(name.strip()) >= 2


SUPPORTED_PLATFORMS = [
    "youtube", "instagram", "tiktok", "twitter", "x", "facebook",
    "linkedin", "twitch", "podcast", "vimeo", "snapchat", "other"
]


def validate_platform(platform: str) -> bool:
    return any(p in platform.lower() for p in SUPPORTED_PLATFORMS)


def normalize_platform(platform: str) -> str:
    for p in SUPPORTED_PLATFORMS:
        if p in platform.lower():
            return p.capitalize()
    return platform.strip().capitalize()


# ---------------------------------------------------------------------------
# Lead State Manager
# ---------------------------------------------------------------------------

class LeadCollector:
    """
    Tracks which lead fields have been collected across conversation turns.
    Used inside the LangGraph state to persist collection progress.
    """

    REQUIRED_FIELDS = ["name", "email", "platform"]

    def __init__(self):
        self.collected: dict = {}
        self.capture_result: Optional[dict] = None
        self.is_captured: bool = False

    def set_field(self, field: str, value: str) -> bool:
        """Validates and stores a field. Returns True if valid."""
        field = field.lower()
        if field == "name" and validate_name(value):
            self.collected["name"] = value.strip().title()
            return True
        elif field == "email" and validate_email(value):
            self.collected["email"] = value.strip().lower()
            return True
        elif field == "platform":
            if validate_platform(value):
                self.collected["platform"] = normalize_platform(value)
                return True
            else:
                # Accept free-text platform as-is
                self.collected["platform"] = value.strip().capitalize()
                return True
        return False

    def missing_fields(self) -> list:
        return [f for f in self.REQUIRED_FIELDS if f not in self.collected]

    def is_complete(self) -> bool:
        return len(self.missing_fields()) == 0

    def next_prompt(self) -> str:
        """Returns the next question to ask the user."""
        missing = self.missing_fields()
        if not missing:
            return ""
        field = missing[0]
        prompts = {
            "name": "Could you share your full name?",
            "email": "What's the best email address to reach you at?",
            "platform": "Which platform do you primarily create content for? (e.g., YouTube, Instagram, TikTok)"
        }
        return prompts.get(field, f"Could you provide your {field}?")

    def execute_capture(self) -> dict:
        """Fires the mock_lead_capture function. Must only be called when complete."""
        if not self.is_complete():
            raise ValueError(f"Cannot capture lead — missing fields: {self.missing_fields()}")
        if self.is_captured:
            return self.capture_result
        self.capture_result = mock_lead_capture(
            name=self.collected["name"],
            email=self.collected["email"],
            platform=self.collected["platform"],
        )
        self.is_captured = True
        return self.capture_result

    def to_dict(self) -> dict:
        return {
            "collected": self.collected,
            "is_captured": self.is_captured,
            "capture_result": self.capture_result,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LeadCollector":
        obj = cls()
        obj.collected = data.get("collected", {})
        obj.is_captured = data.get("is_captured", False)
        obj.capture_result = data.get("capture_result", None)
        return obj


# ---------------------------------------------------------------------------
# LangChain Tool Wrapper (for LangGraph tool node)
# ---------------------------------------------------------------------------

@tool
def capture_lead_tool(name: str, email: str, platform: str) -> str:
    """
    Captures a qualified lead by recording their name, email, and creator platform.
    Only call this tool after ALL three fields have been explicitly confirmed by the user.
    Args:
        name: Full name of the user
        email: Valid email address
        platform: Primary content creation platform (YouTube, Instagram, etc.)
    Returns:
        JSON string with lead capture result
    """
    if not validate_email(email):
        return json.dumps({"status": "error", "message": f"Invalid email: {email}"})
    if not validate_name(name):
        return json.dumps({"status": "error", "message": f"Invalid name: {name}"})

    result = mock_lead_capture(name=name.strip().title(), email=email.strip().lower(), platform=platform)
    return json.dumps(result)
