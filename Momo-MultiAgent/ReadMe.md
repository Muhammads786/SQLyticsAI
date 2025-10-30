
# Quick scaffold

ai-agent/
 app/
     __init__.py
    config.py
    schemas.py
    prompts.py
    tools.py
    memory.py
    agent.py
    multiagent.py
    outputs.py
    evals.py
    api.py
 tests/
    test_contracts.py
    eval_scenarios.jsonl
 .env.example
 requirements.txt
 run.sh




# How the agent works (core logic)

1. **Contract-first I/O**

   * Requests must match `AgentInput` (question, optional context).
   * Responses must match `AgentOutput` (summary_md, findings[], citations[], followups[]).

2. **Context assembly**

   * `app/agent.py` loads any provided URLs/PDFs/text, chunks them, and **upserts** chunks into vector memory.

3. **Memory assist**

   * It queries vector memory for **semantic hits** relevant to the question and injects those snippets into the prompt as extra context.

4. **Prompting & call**

   * Builds a structured user message using `prompts.py` templates and a rules-heavy system prompt.
   * Calls `ChatOpenAI` (or DeepSeek via OpenAI-compatible endpoint).

5. **Strict output parsing**

   * Extracts the JSON block from the model reply and validates it against `AgentOutput` (Pydantic).
   * If schema mismatch, it returns a salvageable minimal output.

6. **(Optional) Multi-agent orchestration**

   * `multiagent.py` runs a tiny 3-node graph (Planner → Researcher → Reporter) that reuses the single agent to refine results.

7. **Delivery**

   * `api.py` exposes `/agent`, `/multi`, and `/agent/pdf`.
   * `outputs.py` converts Markdown to HTML or a simple PDF for exec-ready briefs.

---



# File-by-file: responsibilities, key functions, upgrades

### `app/config.py`

* **What it does:** Simple container for the agent’s role, audience, objectives.
* **Key:** `AgentConfig` (Pydantic model).
* **Enhance:**

  * Load per-tenant config (brand voice, compliance rules).
  * Add toggles for safety levels, temperature, provider/model routing.

### `app/schemas.py`

* **What it does:** The **contracts**.
* **Key:** `AgentInput`, `RetrievalItem`, `Finding`, `AgentOutput`.
* **Enhance:**

  * Add `constraints` (max lengths, enums), richer `Finding` (severity, category).
  * Add `error`/`trace` fields for self-diagnostics.

### `app/prompts.py`

* **What it does:** System rules + human template for consistent outputs.
* **Key:** `SYSTEM_PROMPT`, `HUMAN_TEMPLATE`.
* **Enhance:**

  * Parameterize tone (exec brief vs. engineer detail).
  * Insert **guardrails** (cite-or-say-don’t-know, PII redaction policy).
  * Add few-shot exemplars for tricky tasks.

### `app/tools.py`

* **What it does:** Loaders, chunking, and (stub) search.
* **Key:** `load_pdf`, `fetch_url_text`, `chunk`, `web_search` (stub).
* **Enhance:**

  * Real web search (Serper/Bing), retries, timeouts, MIME detection.
  * Structured readers (HTML boilerplate removal, table extraction).
  * Hash-based dedupe; checksum tags in metadata.

### `app/memory.py`

* **What it does:** Vector store (Chroma) and tiny conversation store.
* **Key:** `upsert_docs`, `search`, `remember`, `recall`.
* **Enhance:**

  * Swap to **Qdrant** (you already use it elsewhere) + namespaces per tenant.
  * Hybrid search (BM25 + embeddings), RAG re-ranking (ColBERT/Splade).
  * TTL and GDPR “forget” APIs; memory governance tags.

### `app/agent.py`  ✅ core logic

* **What it does:** Orchestrates context → memory → prompt → LLM → parse → contract validate.
* **Key path:**

  * `_context_to_text()` loads resources + upserts chunks.
  * `mem_search()` for semantic recall.
  * Builds messages with `prompts.py`.
  * Calls `ChatOpenAI`, extracts JSON with regex, validates with `AgentOutput`.
* **Enhance:**

  * **Deterministic output parsing** (LC Output Parsers / PydanticOutputParser).
  * **Toolformer** pattern: if citations missing, auto-trigger web search step.
  * **Streaming** tokens to UI; function-calling for tool execution.
  * **Safety**: profanity/PII filters, domain allowlists.
  * **Caching**: request/response + embedding cache (FNV-1a of chunks).
  * **Observability**: LangSmith/OpenTelemetry traces, request IDs.

### `app/multiagent.py`

* **What it does:** Minimal **LangGraph** with Planner → Researcher → Reporter state machine.
* **Key:** `State`, node functions, `build_graph()`.
* **Enhance:**

  * Add a **ToolRouter** node (retrieval, calc, code-execute).
  * Branch & retry policy (if low confidence → re-plan).
  * Memory write-backs after finalization (long-term learnings).

### `app/outputs.py`

* **What it does:** Rendering utilities.
* **Key:** `md_to_html`, `md_to_pdf`.
* **Enhance:**

  * Rich PDF templates (headers, logos, tables, footnotes).
  * HTML templates with Tailwind + assets; export to DOCX.
  * Attachments packager (zip JSON + PDF + sources).

### `app/api.py`

* **What it does:** Productizes the agent.
* **Key:** `/agent` (single), `/multi` (graph), `/agent/pdf` (render).
* **Enhance:**

  * Auth (JWT/API keys), **rate limiting**, request size caps.
  * Error mapping (LLM timeouts, provider errors) to stable codes.
  * Audit logs + per-tenant quotas; CORS policy.

### `app/evals.py`

* **What it does:** Quick smoke tests with keyword hit-rate + latency.
* **Key:** `run_eval()`, `keyword_hitrate()`.
* **Enhance:**

  * **RAGAS**/custom rubrics, golden-truth scoring, BLEURT/BERTScore.
  * Regression suite per release; CI gate (fail on drift).
  * Cost/latency dashboards.

### `tests/test_contracts.py`

* **What it does:** Verifies schema adherence.
* **Enhance:**

  * Add property-based tests (hypothesis) for edge inputs.
  * Snapshot tests for stable prompts; contract tests for tools.

---

# Mermaid — component architecture


```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant API as FastAPI<br/>/agent
    participant AG as Agent<br/>run_agent()
    participant CTX as Context Tools<br/>Loaders/Chunker
    participant VM as Vector Memory<br/>ChromaDB
    participant LLM as LLM Provider<br/>ChatOpenAI/DeepSeek
    participant OUT as Output Renderer<br/>MD→PDF/HTML

    Note over U,OUT: 🚀 Agent Query Initiation
    U->>API: POST /agent {question, context[]}
    API->>AG: run_agent(payload)
    
    Note over AG,VM: 🔍 Memory Retrieval Phase
    AG->>VM: mem_search(question)
    VM-->>AG: top-k relevant snippets
    
    Note over AG,CTX: 📥 Context Enrichment Phase
    AG->>CTX: fetch_url_text() / load_pdf() + chunk()
    CTX-->>AG: processed text chunks
    
    Note over AG,VM: 💾 Memory Storage Phase
    AG->>VM: upsert(chunks + metadata)
    
    Note over AG,LLM: 🤖 LLM Reasoning Phase
    AG->>LLM: system + user prompt<br/>(with memory + context)
    LLM-->>AG: markdown + ```json``` block
    
    Note over AG: ⚙️ Output Processing Phase
    AG->>AG: parse_json() + validate_output()
    AG->>AG: create AgentOutput object
    
    Note over API,OUT: 📤 Response Delivery Phase
    AG-->>API: AgentOutput
    API-->>U: JSON response
    
    Note over API,OUT: 🎨 Optional Format Conversion
    API->>OUT: convert_markdown_to_pdf()
    OUT-->>API: rendered PDF/HTML
    API-->>U: formatted document (optional)

    %% Dark Theme Styling
    classDef actor fill:#1a237e,stroke:#3949ab,color:#e3f2fd
    classDef api fill:#006064,stroke:#00838f,color:#e0f2f1
    classDef agent fill:#4a148c,stroke:#7b1fa2,color:#f3e5f5
    classDef tools fill:#33691e,stroke:#558b2f,color:#f1f8e9
    classDef memory fill:#bf360c,stroke:#e64a19,color:#fbe9e7
    classDef llm fill:#880e4f,stroke:#c2185b,color:#fce4ec
    classDef output fill:#5d4037,stroke:#795548,color:#efebe9
    classDef note fill:#1b5e20,stroke:#388e3c,color:#e8f5e8

    class U actor
    class API api
    class AG agent
    class CTX tools
    class VM memory
    class LLM llm
    class OUT output
    class Note note
```

## 🎨 **Dark Theme Color Scheme**

### **👤 User (Actor)** - Deep Blue
- System initiator with clear visual distinction

### **🔗 API Layer** - Teal 
- FastAPI endpoint handling requests/responses

### **🤖 Agent Core** - Purple
- Central orchestration and business logic

### **🛠️ Context Tools** - Dark Green
- Data processing, loading, and chunking utilities

### **🧠 Vector Memory** - Red-Orange
- ChromaDB for semantic search and storage

### **💬 LLM Provider** - Pink
- AI model interactions (OpenAI/DeepSeek)

### **📄 Output Renderer** - Brown
- Format conversion and document generation

### **📝 Process Notes** - Green
- Phase descriptions and workflow milestones

## 🔄 **Workflow Phase Breakdown**

### **1. 🚀 Query Initiation** (Steps 1-2)
- User submits question with optional context
- FastAPI routes to agent execution

### **2. 🔍 Memory Retrieval** (Steps 3-4)  
- Semantic search in vector database
- Retrieves relevant historical context

### **3. 📥 Context Enrichment** (Steps 5-6)
- Fetches and processes external content
- Chunks data for optimal processing

### **4. 💾 Memory Storage** (Step 7)
- Stores new context with metadata
- Updates knowledge base

### **5. 🤖 LLM Reasoning** (Steps 8-9)
- Enhanced prompt with full context
- Structured JSON + markdown generation

### **6. ⚙️ Output Processing** (Steps 10-11)
- JSON parsing and validation
- AgentOutput object creation

### **7. 📤 Response Delivery** (Steps 12-13)
- JSON response to user
- Optional formatted document generation



---

# High-impact enhancements (prioritized)

1. **Output robustness**

   * Replace regex JSON extraction with LangChain’s `PydanticOutputParser` or `tool/response` schema calls.
   * Add a **repair loop**: if validation fails, ask the model to fix to schema.

2. **RAG quality**

   * Switch to **Qdrant** with HNSW + scalar filters; add **re-ranking** (cross-encoder).
   * Source-grounding: require citations and verify URLs are present in memory.

3. **Observability & safety**

   * OpenTelemetry traces (request→LLM calls→tools).
   * PII detector + redactor; domain allowlists for web pulls.

4. **Performance & cost**

   * Embedding + response cache keyed by (prompt hash, chunk hashes).
   * Batch embeddings, async loaders, timeouts/retries with backoff.

5. **Multi-agent orchestration**

   * Confidence-driven re-planning, specialized tools per node (Search, Calculator, CodeExec).
   * Guarded tool execution sandbox.

6. **Productization**

   * AuthN/AuthZ, per-tenant config & memory namespaces, billing hooks.
   * Streaming responses (server-sent events) to UI.


