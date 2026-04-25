"""
AutoStream Agent — Test Suite
Tests: RAG pipeline, intent classifier, lead collector, and agent graph.
Run with: pytest tests/ -v
"""

import pytest
import json
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# RAG Pipeline Tests
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    def setup_method(self):
        from agent.rag_pipeline import RAGPipeline
        self.rag = RAGPipeline()

    def test_loads_documents(self):
        assert len(self.rag.documents) > 0

    def test_retrieves_pricing_for_price_query(self):
        results = self.rag.retrieve("how much does the Pro plan cost", top_k=3)
        assert len(results) > 0
        texts = " ".join([r[0] for r in results]).lower()
        assert "pro" in texts or "79" in texts

    def test_retrieves_basic_plan_info(self):
        results = self.rag.retrieve("basic plan features 720p", top_k=3)
        texts = " ".join([r[0] for r in results]).lower()
        assert "basic" in texts or "29" in texts or "720" in texts

    def test_retrieves_refund_policy(self):
        results = self.rag.retrieve("refund policy", top_k=3)
        texts = " ".join([r[0] for r in results]).lower()
        assert "refund" in texts

    def test_retrieves_support_policy(self):
        results = self.rag.retrieve("24/7 support", top_k=3)
        texts = " ".join([r[0] for r in results]).lower()
        assert "support" in texts

    def test_context_string_not_empty(self):
        ctx = self.rag.get_context_string("pricing plans")
        assert ctx and len(ctx) > 10

    def test_no_results_for_nonsense(self):
        results = self.rag.retrieve("xyzzy foobar gibberish", top_k=3)
        # Score should be 0 for all, so list may be empty
        assert isinstance(results, list)

    def test_top_k_respected(self):
        results = self.rag.retrieve("plan features", top_k=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Intent Classifier Tests
# ---------------------------------------------------------------------------

class TestIntentClassifier:
    def setup_method(self):
        from agent.intent_classifier import classify_intent_heuristic, Intent
        self.classify = classify_intent_heuristic
        self.Intent = Intent

    def test_greeting_hi(self):
        assert self.classify("Hi there!") == self.Intent.GREETING

    def test_greeting_hello(self):
        assert self.classify("Hello, good morning") == self.Intent.GREETING

    def test_product_inquiry_pricing(self):
        result = self.classify("What are your pricing plans?")
        assert result == self.Intent.PRODUCT_INQUIRY

    def test_product_inquiry_features(self):
        result = self.classify("Tell me about the features of the Pro plan")
        assert result == self.Intent.PRODUCT_INQUIRY

    def test_high_intent_sign_up(self):
        result = self.classify("I want to sign up for the Pro plan")
        assert result == self.Intent.HIGH_INTENT_LEAD

    def test_high_intent_buy(self):
        result = self.classify("I want to buy this for my YouTube channel")
        assert result == self.Intent.HIGH_INTENT_LEAD

    def test_high_intent_ready_to_sign(self):
        result = self.classify("That sounds great, I'm ready to sign up")
        assert result == self.Intent.HIGH_INTENT_LEAD

    def test_high_intent_try(self):
        result = self.classify("I want to try the service, sign me up")
        assert result == self.Intent.HIGH_INTENT_LEAD

    def test_product_refund(self):
        result = self.classify("What is your refund policy?")
        assert result == self.Intent.PRODUCT_INQUIRY

    def test_product_support(self):
        result = self.classify("Do you have 24/7 support?")
        assert result == self.Intent.PRODUCT_INQUIRY


# ---------------------------------------------------------------------------
# Lead Collector Tests
# ---------------------------------------------------------------------------

class TestLeadCollector:
    def setup_method(self):
        from tools.lead_capture import LeadCollector
        self.collector = LeadCollector()

    def test_initially_empty(self):
        assert self.collector.missing_fields() == ["name", "email", "platform"]
        assert not self.collector.is_complete()

    def test_set_valid_name(self):
        assert self.collector.set_field("name", "John Doe")
        assert self.collector.collected["name"] == "John Doe"

    def test_set_valid_email(self):
        assert self.collector.set_field("email", "john@example.com")
        assert self.collector.collected["email"] == "john@example.com"

    def test_set_invalid_email(self):
        result = self.collector.set_field("email", "not-an-email")
        assert not result
        assert "email" not in self.collector.collected

    def test_set_platform(self):
        assert self.collector.set_field("platform", "YouTube")
        assert "platform" in self.collector.collected

    def test_missing_fields_updates_on_set(self):
        self.collector.set_field("name", "Alice Smith")
        missing = self.collector.missing_fields()
        assert "name" not in missing
        assert "email" in missing

    def test_is_complete_after_all_fields(self):
        self.collector.set_field("name", "Alice Smith")
        self.collector.set_field("email", "alice@test.com")
        self.collector.set_field("platform", "YouTube")
        assert self.collector.is_complete()

    def test_execute_capture_when_complete(self):
        self.collector.set_field("name", "Bob Jones")
        self.collector.set_field("email", "bob@test.com")
        self.collector.set_field("platform", "Instagram")
        result = self.collector.execute_capture()
        assert result["status"] == "success"
        assert "lead_id" in result
        assert self.collector.is_captured

    def test_execute_capture_raises_if_incomplete(self):
        self.collector.set_field("name", "Incomplete User")
        with pytest.raises(ValueError):
            self.collector.execute_capture()

    def test_next_prompt_name_first(self):
        prompt = self.collector.next_prompt()
        assert "name" in prompt.lower()

    def test_next_prompt_email_after_name(self):
        self.collector.set_field("name", "Test User")
        prompt = self.collector.next_prompt()
        assert "email" in prompt.lower()

    def test_serialization_roundtrip(self):
        from tools.lead_capture import LeadCollector
        self.collector.set_field("name", "Roundtrip User")
        self.collector.set_field("email", "rt@test.com")
        data = self.collector.to_dict()
        restored = LeadCollector.from_dict(data)
        assert restored.collected["name"] == "Roundtrip User"
        assert restored.collected["email"] == "rt@test.com"
        assert "platform" in restored.missing_fields()


# ---------------------------------------------------------------------------
# Field Extraction Tests
# ---------------------------------------------------------------------------

class TestFieldExtraction:
    def setup_method(self):
        from agent.nodes import _extract_field_from_message
        from tools.lead_capture import LeadCollector
        self.extract = _extract_field_from_message
        self.collector = LeadCollector()

    def test_extract_email(self):
        val = self.extract("Sure, my email is jane@gmail.com", "email", self.collector)
        assert val == "jane@gmail.com"

    def test_extract_email_from_short_reply(self):
        val = self.extract("test.user@domain.co.uk", "email", self.collector)
        assert val == "test.user@domain.co.uk"

    def test_extract_platform_youtube(self):
        val = self.extract("I mainly use YouTube", "platform", self.collector)
        assert val is not None
        assert "youtube" in val.lower()

    def test_extract_platform_instagram(self):
        val = self.extract("Instagram is my main platform", "platform", self.collector)
        assert val is not None
        assert "instagram" in val.lower()

    def test_extract_name_short_phrase(self):
        val = self.extract("John Smith", "name", self.collector)
        assert val is not None

    def test_extract_name_from_sentence(self):
        val = self.extract("My name is Sarah Connor", "name", self.collector)
        assert val is not None
        assert "Sarah" in val


# ---------------------------------------------------------------------------
# Mock Lead Capture Function Tests
# ---------------------------------------------------------------------------

class TestMockLeadCapture:
    def test_mock_lead_capture_returns_dict(self):
        from tools.lead_capture import mock_lead_capture
        result = mock_lead_capture("Test User", "test@example.com", "YouTube")
        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["name"] == "Test User"
        assert result["email"] == "test@example.com"
        assert result["platform"] == "YouTube"
        assert "lead_id" in result
        assert "captured_at" in result

    def test_mock_lead_capture_unique_ids(self):
        from tools.lead_capture import mock_lead_capture
        r1 = mock_lead_capture("User A", "a@test.com", "YouTube")
        r2 = mock_lead_capture("User B", "b@test.com", "Instagram")
        assert r1["lead_id"] != r2["lead_id"]


# ---------------------------------------------------------------------------
# Validators Tests
# ---------------------------------------------------------------------------

class TestValidators:
    def test_valid_emails(self):
        from tools.lead_capture import validate_email
        assert validate_email("user@example.com")
        assert validate_email("user.name+tag@domain.co.uk")
        assert validate_email("test123@subdomain.example.org")

    def test_invalid_emails(self):
        from tools.lead_capture import validate_email
        assert not validate_email("notanemail")
        assert not validate_email("missing@")
        assert not validate_email("@nodomain.com")
        assert not validate_email("")

    def test_valid_names(self):
        from tools.lead_capture import validate_name
        assert validate_name("Alice")
        assert validate_name("John Doe")
        assert validate_name("María García")

    def test_invalid_names(self):
        from tools.lead_capture import validate_name
        assert not validate_name("A")
        assert not validate_name("")
        assert not validate_name(" ")


# ---------------------------------------------------------------------------
# Session Manager Tests
# ---------------------------------------------------------------------------

class TestSessionManager:
    def setup_method(self):
        from utils.session_manager import SessionStore
        self.store = SessionStore()

    def test_get_nonexistent_returns_none(self):
        assert self.store.get("nonexistent_session") is None

    def test_set_and_get(self):
        self.store.set("session_1", {"messages": [], "turn_count": 1})
        result = self.store.get("session_1")
        assert result is not None
        assert result["turn_count"] == 1

    def test_delete(self):
        self.store.set("session_2", {"messages": []})
        self.store.delete("session_2")
        assert self.store.get("session_2") is None

    def test_active_sessions_count(self):
        self.store.set("s1", {"messages": []})
        self.store.set("s2", {"messages": []})
        assert self.store.active_sessions() >= 2
