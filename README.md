#**Conversational Lead Generation AI Agent** 🎬
> Social-to-Lead Agentic Workflow | Built with LangGraph + RAG + Tool Calling

A production-grade conversational AI sales agent for **AutoStream** — a SaaS platform offering automated video editing tools for content creators. The agent qualifies leads through natural conversation, retrieves accurate product knowledge via RAG, and captures user details via a mock CRM API.

---

## Project Structure

```
autostream-agent/
├── agent/
│   ├── graph.py             # LangGraph graph builder + AutoStreamAgent wrapper
│   ├── nodes.py             # All LangGraph node implementations
│   ├── intent_classifier.py # Heuristic + LLM-based intent classification
│   ├── rag_pipeline.py      # TF-IDF RAG over local JSON knowledge base
│   └── state.py             # AgentState TypedDict schema
├── tools/
│   └── lead_capture.py      # LeadCollector, mock_lead_capture, LangChain tool
├── utils/
│   ├── logger.py            # Coloured structured logger
│   └── session_manager.py   # Per-user session store (in-memory, Redis-ready)
├── knowledge_base/
│   └── autostream_kb.json   # Pricing, features, and policies knowledge base
├── tests/
│   └── test_agent.py        # Pytest test suite (35+ tests)
├── main.py                  # CLI entrypoint
├── webhook_server.py        # FastAPI server (REST + WhatsApp webhook)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### 1. Clone & Install

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Open `.env` and set your LLM API key. At minimum:

```env
ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY / GOOGLE_API_KEY
LLM_PROVIDER=anthropic
```

### 3. Run CLI (Recommended for Demo)

```bash
python main.py
# With a specific provider:
python main.py --provider openai
python main.py --provider google
```

**CLI commands during session:**
- `debug` — print current intent, lead state, turn count
- `reset` — start a new session
- `quit` / `exit` — end session

### 4. Run Webhook Server

```bash
uvicorn webhook_server:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Generic chat API |
| `POST` | `/chat/reset?session_id=X` | Reset a session |
| `GET` | `/whatsapp` | Meta webhook verification |
| `POST` | `/whatsapp` | Incoming WhatsApp messages |
| `GET` | `/metrics` | Active sessions + config |

**Test the REST API:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user_001", "message": "Hi, what are your pricing plans?"}'
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## Architecture Explanation (~200 words)

The agent is built on **LangGraph**, a stateful graph execution framework built on top of LangChain. LangGraph was chosen over AutoGen because it provides explicit, inspectable state management through a typed `AgentState` schema — making the flow predictable, debuggable, and production-deployable.

**Graph Flow:**
1. **`classify_intent`** — Every user message is passed through a two-stage classifier: fast keyword heuristics first, then an LLM call for ambiguous inputs. This keeps latency low on simple cases.
2. **`retrieve_context`** — For product queries, a TF-IDF RAG pipeline retrieves the top-3 relevant chunks from the local JSON knowledge base. No external vector DB is needed; the KB is small and deterministic.
3. **`activate_lead` → `handle_lead_collection`** — When high intent is detected, the graph transitions into lead collection mode. A `LeadCollector` object (serialized into state) progressively gathers name, email, and platform — one field per turn — before calling `mock_lead_capture()`.
4. **`generate_response`** — Greeting and product inquiry messages invoke the LLM with a RAG-augmented system prompt.

**State** is a `TypedDict` (`AgentState`) passed immutably between nodes. LangGraph's `add_messages` reducer handles message accumulation, giving the agent full multi-turn memory across 5–6+ conversation turns without external storage.

---

## WhatsApp Webhook Integration

### How It Works

WhatsApp (Meta Cloud API) sends incoming messages as HTTP POST requests to your server. Here's the end-to-end flow:

```
User (WhatsApp) → Meta Cloud API → POST /whatsapp → AutoStream Agent → POST graph.facebook.com → User (WhatsApp)
```

### Step-by-Step Setup

**1. Create a Meta App**
- Go to [developers.facebook.com](https://developers.facebook.com)
- Create a new App → Business type → Add WhatsApp product

**2. Configure the Webhook**
- In WhatsApp → Configuration → Webhook URL: `https://your-domain.com/whatsapp`
- Verify Token: set the same value as `WHATSAPP_VERIFY_TOKEN` in your `.env`
- Subscribe to: `messages`

**3. Expose Your Server**
```bash
# For local development, use ngrok:
ngrok http 8000
# Use the HTTPS URL ngrok gives you as your webhook URL
```

**4. Set Environment Variables**
```env
WHATSAPP_VERIFY_TOKEN=autostream_verify_2024
WHATSAPP_APP_SECRET=<from Meta App Dashboard>
WHATSAPP_ACCESS_TOKEN=<temporary or permanent token from Meta>
WHATSAPP_PHONE_NUMBER_ID=<from Meta → WhatsApp → API Setup>
```

**5. Security**
Every incoming POST from Meta is signed with HMAC-SHA256 using your App Secret. The `_verify_whatsapp_signature()` function in `webhook_server.py` validates this before processing any message.

**Session Isolation:** Each WhatsApp sender (`from` number) gets its own isolated session via `session_id = f"wa_{phone_number}"`, preserving full conversation context per user.

---

## Conversation Flow Example

```
You:  Hi there!
Aria: Hello! Welcome to AutoStream — AI-powered video editing for creators...

You:  What does the Pro plan include?
Aria: The Pro plan is $79/month and includes unlimited videos, 4K export,
      AI captions, 24/7 support, and 500GB cloud storage...

You:  That sounds great. I want to try the Pro plan for my YouTube channel.
Aria: Awesome! I'd love to get you started. Could you share your full name?

You:  Alex Johnson
Aria: Great, Alex! What's the best email address to reach you at?

You:  alex@gmail.com
Aria: Perfect! Which platform do you primarily create for?

You:  YouTube
Aria: You're all set, Alex! I've registered your interest in the Pro plan...
      [Lead captured: LEAD-04291 | alex@gmail.com | YouTube]
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| TF-IDF RAG (no vector DB) | KB is small and static; avoids Chroma/Pinecone overhead |
| Heuristic + LLM intent | Fast on clear signals, accurate on ambiguous ones |
| Field-by-field collection | More natural UX; avoids overwhelming users with forms |
| LangGraph over chains | Explicit state machine; easier to extend and debug |
| Serialized LeadCollector | State survives graph re-invocation across turns |
| In-memory sessions (Redis-ready) | Simple default; swap `SessionStore` backend for production |

---

## License

MIT — built for the ServiceHive / Inflx ML Internship Assignment.
