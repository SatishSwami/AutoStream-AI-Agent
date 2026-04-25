"""
RAG Pipeline for AutoStream Agent
Loads local JSON knowledge base, builds a simple vector store using TF-IDF,
and retrieves relevant context for a given user query.
"""

import json
import os
import math
from typing import List, Tuple
from pathlib import Path


KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"


def _load_knowledge_base() -> dict:
    with open(KB_PATH, "r") as f:
        return json.load(f)


def _flatten_kb_to_documents(kb: dict) -> List[dict]:
    """
    Convert the nested KB JSON into a flat list of retrievable documents.
    Each doc has: {'id': str, 'text': str, 'source': str}
    """
    docs = []

    # Company overview
    company = kb["company"]
    docs.append({
        "id": "company_overview",
        "text": f"{company['name']}: {company['tagline']}. {company['description']}",
        "source": "company"
    })

    # Pricing plans
    for plan in kb["pricing"]["plans"]:
        features_text = ", ".join(plan["features"])
        price_text = (
            f"{plan['name']} Plan costs ${plan['price_monthly']}/month "
            f"(or ${plan['price_annual']}/year). "
            f"Features include: {features_text}. "
            f"Ideal for: {plan['ideal_for']}."
        )
        docs.append({
            "id": f"pricing_{plan['name'].lower()}",
            "text": price_text,
            "source": "pricing"
        })

    # Policies
    for policy in kb["policies"]:
        docs.append({
            "id": policy["id"],
            "text": f"{policy['title']}: {policy['content']}",
            "source": "policy"
        })

    # FAQs
    for i, faq in enumerate(kb["faqs"]):
        docs.append({
            "id": f"faq_{i}",
            "text": f"Q: {faq['question']} A: {faq['answer']}",
            "source": "faq"
        })

    return docs


def _tokenize(text: str) -> List[str]:
    return text.lower().replace(",", " ").replace(".", " ").replace("?", " ").split()


def _compute_tf(tokens: List[str]) -> dict:
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    total = len(tokens)
    return {k: v / total for k, v in tf.items()}


def _compute_idf(docs_tokens: List[List[str]]) -> dict:
    N = len(docs_tokens)
    idf = {}
    all_tokens = set(token for doc in docs_tokens for token in doc)
    for token in all_tokens:
        df = sum(1 for doc in docs_tokens if token in doc)
        idf[token] = math.log((N + 1) / (df + 1)) + 1
    return idf


def _tfidf_score(query_tokens: List[str], doc_tf: dict, idf: dict) -> float:
    score = 0.0
    for token in query_tokens:
        if token in doc_tf and token in idf:
            score += doc_tf[token] * idf[token]
    return score


class RAGPipeline:
    """
    Lightweight RAG pipeline using TF-IDF similarity over a local JSON knowledge base.
    No external vector DB required — fully self-contained.
    """

    def __init__(self):
        kb = _load_knowledge_base()
        self.documents = _flatten_kb_to_documents(kb)
        docs_tokens = [_tokenize(doc["text"]) for doc in self.documents]
        self.idf = _compute_idf(docs_tokens)
        self.doc_tfs = [_compute_tf(tokens) for tokens in docs_tokens]

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve top-k relevant documents for a given query.
        Returns list of (document_text, score) tuples.
        """
        query_tokens = _tokenize(query)
        scores = [
            (doc["text"], _tfidf_score(query_tokens, tf, self.idf), doc["source"])
            for doc, tf in zip(self.documents, self.doc_tfs)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(text, score, source) for text, score, source in scores[:top_k] if score > 0]

    def get_context_string(self, query: str, top_k: int = 3) -> str:
        """
        Returns a formatted context string ready to be injected into an LLM prompt.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, (text, score, source) in enumerate(results, 1):
            context_parts.append(f"[Source: {source}]\n{text}")

        return "\n\n".join(context_parts)


# Singleton instance
_rag_pipeline: RAGPipeline = None


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
