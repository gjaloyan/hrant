# SELF-LEARNING AGENT — Full Build Specification for Claude Code

## PROJECT OVERVIEW

Build a **Self-Learning AI Agent** — a local-first application where the AI model does NOT store domain knowledge in its weights. Instead, it learns on-demand by reading sources, creating structured notes, and storing them locally. The agent grows smarter over time on specific tasks/projects while keeping a small, efficient core.

Think of it as: **a junior engineer with perfect note-taking skills who never forgets a notebook.**

---

## ARCHITECTURE (4 Layers)

### Layer 0: LLM Core (Brain)
- Use Anthropic Claude API (`claude-sonnet-4-20250514`) as the reasoning engine
- Can be swapped for local models (Ollama with Mistral/Llama/Phi) via config
- The core ONLY provides: language understanding, logical reasoning, ability to learn, basic world knowledge
- The core does NOT store domain-specific knowledge — that lives in Layer 2

### Layer 1: Orchestrator (Agent Loop)
- Task Analyzer: decompose user request → required knowledge topics → action plan
- Knowledge Router: for each topic → check local memory → if missing → learn it
- Tool Router: decide which tool to use (web search, file read, code execution, memory write)
- Self-Verifier: after generating answer → verify every claim against source notes
- Confidence Reporter: output confidence % and list unverified claims

### Layer 2: Local Memory (Knowledge Base)
Directory structure:
```
knowledge/
├── core_memory.md          # ALWAYS loaded into context (permanent facts)
├── index.json              # Map of all topics: keywords, file paths, access counts
├── access_log.json         # Track how often each topic is accessed
├── finetune_queue.jsonl    # Pairs for future model fine-tuning
├── fundamentals/           # School-level: physics, chemistry, math, etc.
├── profession/             # Domain expertise: RS-485, PLCs, Modbus, etc.
├── projects/               # Project-specific context and decisions
│   └── project_name/
│       ├── overview.md
│       ├── decisions.md    # Why we chose X over Y
│       ├── issues.md       # Known bugs, workarounds
│       └── hardware.md     # Specific equipment on this project
└── personal/               # User preferences, work style, contacts
```

Note format (every .md file):
```markdown
---
topic: RS-485
category: profession
created: 2025-01-15 14:30
updated: 2025-01-20 09:15
keywords: rs485, serial, modbus, differential, bus
source: https://example.com/rs485-spec
confidence: verified
access_count: 15
---

# RS-485

## What It Is
(concise, factual, only key information)

## Key Parameters
(numbers, specs, limits)

## Practical Notes
(real-world tips from experience)

## Common Mistakes
(things that went wrong and fixes)

## Related Topics
- [[MAX485]] — transceiver IC
- [[Modbus RTU]] — protocol layer
```

### Layer 3: Tools
- **Web Search**: find documentation, datasheets, articles (use Anthropic web search tool or SerpAPI or Tavily)
- **File Reader**: read PDFs, DOCX, images, datasheets uploaded by user
- **Code Executor**: run Python/JS to test hypotheses, calculate, validate
- **Memory Writer**: create/update/delete notes in knowledge base
- **Memory Search**: semantic search across all notes (use ChromaDB or simple keyword matching)

---

## CORE AGENT LOOP (implement this exactly)

```
USER TASK
    │
    ▼
┌─────────────────────┐
│  1. LOAD CORE MEMORY │  ← Always load core_memory.md into context
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. ANALYZE TASK     │  ← LLM determines what knowledge is needed
│     Output:          │     Returns: required_topics[], plan[], confidence
│     - required topics│
│     - action plan    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────┐
│  3. KNOWLEDGE CHECK & LEARN  │  ← For each required topic:
│     For each topic:          │
│     ├─ Search local memory   │     a) Search index.json by keywords
│     ├─ IF found → load note  │     b) If found → load into context
│     └─ IF not found:         │     c) If NOT found:
│        ├─ Search web/docs    │        - Search web for information
│        ├─ Read sources       │        - Read and extract key facts
│        ├─ Create note (md)   │        - Write structured note to disk
│        ├─ Update index       │        - Update index.json
│        └─ Load into context  │        - Load new note into context
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────┐
│  4. SOLVE TASK       │  ← LLM solves task using ONLY loaded notes
│     Rules:           │     System prompt: "Answer ONLY from notes.
│     - Only use notes │      If info is missing, say what's missing.
│     - No guessing    │      Never fabricate facts."
│     - Cite sources   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  5. SELF-VERIFY      │  ← Second LLM pass checks answer against notes
│     Check:           │     For each claim in answer:
│     - Each claim     │     - Is it in the notes? → verified
│     - Hallucinations │     - Not in notes? → flag as unverified
│     - Confidence %   │     - Contradicts notes? → flag as error
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────┐
│  6. LEARN FROM EXPERIENCE    │
│     - Log access counts      │  ← Track which topics are used most
│     - Auto-promote to core?  │  ← If topic accessed 10+ times → suggest core
│     - Add to finetune queue? │  ← Save good Q&A pairs for future training
│     - Update notes with      │  ← If task revealed new info → update note
│       new discoveries        │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────┐
│  7. CLEANUP          │  ← Unload notes from context (they stay on disk)
│     - Free context   │     Clear working memory
│     - Notes persist  │     Knowledge base grows permanently
└─────────────────────┘
```

---

## THREE LEVELS OF MEMORY (implement all three)

### Level 1: Core Memory (always in context)
- File: `knowledge/core_memory.md`
- Loaded into system prompt at EVERY agent run
- User can say: "remember this forever" → agent adds to core_memory.md
- Auto-promotion: if a topic is accessed 10+ times → agent suggests promoting key facts to core
- Should stay small (< 4000 tokens) to not waste context window
- Commands: `remember <fact>`, `forget <fact>`, `show core memory`

### Level 2: Knowledge Notes (loaded on demand)
- Files: `knowledge/{category}/{topic}.md`
- Searched by keywords via index.json
- Loaded into context only when needed for current task
- Unloaded after task completion
- Agent creates these automatically when learning new topics
- Commands: `learn about <topic>`, `what do you know about <topic>`, `show all topics`

### Level 3: Fine-tune Queue (for model improvement)
- File: `knowledge/finetune_queue.jsonl`
- Every verified good answer is saved as instruction→response pair
- When 50+ examples accumulated → can export for fine-tuning
- Compatible with OpenAI/HuggingFace fine-tune format
- Commands: `finetune status`, `export finetune data`

---

## USER INTERACTION COMMANDS

The agent should understand these natural language commands:

### Memory Management
- "Remember that [fact]" → add to core_memory.md
- "Forget about [fact]" → remove from core_memory.md
- "Learn about [topic]" → research topic, create note
- "What do you know?" → list all topics in knowledge base
- "What do you know about [topic]?" → show note content
- "Show core memory" → display permanent facts
- "Delete knowledge about [topic]" → remove note from disk

### Project Mode
- "Start project [name]" → create project folder in knowledge/projects/
- "Project context: [info]" → save to project overview
- "We decided [decision] because [reason]" → log to decisions.md
- "Issue: [problem] → Fix: [solution]" → log to issues.md
- "End project" → archive, suggest key learnings for core memory

### Learning Control
- "Learn this deeply" → create detailed note with examples
- "Quick note: [fact]" → short note, minimal research
- "Verify your answer" → force self-verification pass
- "How confident are you?" → show verification breakdown
- "Learn from this mistake" → update notes with correction

---

## TECH STACK

### Backend
- **Python 3.11+**
- **FastAPI** — REST API server
- **httpx** — async HTTP client for LLM API calls
- **ChromaDB** — vector search for semantic note lookup (optional, can start with keyword search)
- **Pydantic** — data models and validation

### Frontend
- **React** with TypeScript
- **Tailwind CSS** for styling
- **Minimal UI**: chat interface + sidebar showing knowledge base status

### LLM Integration — DUAL MODEL ARCHITECTURE
The agent uses TWO models with different roles:

**Model A — Claude Sonnet (API) — The Brain:**
- Used for: reasoning, task analysis, solving, verification, note creation
- Model: `claude-sonnet-4-20250514` via Anthropic API
- Why: best reasoning quality, understands Russian/English/Armenian
- Cost: pay per API call, but agent minimizes calls via caching
- When offline or API unavailable: fallback to local Qwen

**Model B — Qwen 2.5 7B (Local via Ollama) — The Apprentice:**
- Used for: simple queries when answer exists in notes, keyword extraction, note search, quick lookups
- Also: THE model that gets fine-tuned with collected experience data
- Model: `qwen2.5:7b` via Ollama (runs locally, free, private)
- Why: same architecture family as larger Qwen models, excellent multilingual support, small enough to fine-tune on consumer GPU (RTX 3060 12GB)
- Over time: as Qwen 7B gets fine-tuned on project data, it handles more tasks locally without needing Claude API calls

**Smart Routing Logic:**
```
User task arrives
    │
    ├─ Simple lookup (answer is in notes verbatim)
    │   └─→ Qwen 7B local (fast, free, private)
    │
    ├─ Complex reasoning, new learning, verification
    │   └─→ Claude Sonnet API (smart, accurate)
    │
    ├─ API unavailable / offline mode
    │   └─→ Qwen 7B local (degraded but functional)
    │
    └─ After fine-tuning (v1, v2, v3...)
        └─→ Qwen 7B handles increasingly complex tasks locally
            Claude API needed less and less over time
```

**The Evolution Path:**
- Month 1: 90% Claude API / 10% local Qwen → expensive but accurate
- Month 3: 60% Claude / 40% Qwen (fine-tuned v1) → cheaper
- Month 6: 30% Claude / 70% Qwen (fine-tuned v3) → mostly local
- Month 12: 10% Claude / 90% Qwen (fine-tuned v5+) → almost free
  Claude only needed for genuinely novel problems

### Storage
- **Local filesystem** — markdown files + JSON index
- No database required for MVP
- Optional: SQLite for metadata if scaling
- **Ollama** — local model runtime (install: https://ollama.com)
  - `ollama pull qwen2.5:7b` — download base model
  - Fine-tuned versions registered automatically by pipeline

---

## FILE STRUCTURE

```
self-learning-agent/
├── README.md
├── .env.example                 # ANTHROPIC_API_KEY=
├── config.yaml                  # Model selection, paths, thresholds
├── requirements.txt
│
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── agent.py                 # Main agent orchestrator loop
│   ├── llm.py                   # DUAL MODEL ROUTER (Claude + Qwen)
│   ├── knowledge_manager.py     # CRUD for knowledge notes
│   ├── core_memory.py           # Core memory management
│   ├── note_creator.py          # Structured note generation from sources
│   ├── searcher.py              # Knowledge search (keyword + optional vector)
│   ├── verifier.py              # Self-verification module (always uses Claude)
│   ├── finetune_pipeline.py     # Collect data → curate → train → register
│   ├── model_evaluator.py       # Compare old vs new model before switching
│   ├── tools/
│   │   ├── web_search.py        # Web search tool
│   │   ├── file_reader.py       # PDF/DOCX/image reader
│   │   └── code_executor.py     # Sandboxed code execution
│   └── models.py                # Pydantic models
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Chat.tsx         # Main chat interface
│   │   │   ├── KnowledgePanel.tsx  # Sidebar: topics, core memory
│   │   │   ├── NoteViewer.tsx   # View/edit individual notes
│   │   │   └── StatusBar.tsx    # Confidence, verification status
│   │   └── api.ts               # Backend API calls
│   └── package.json
│
├── knowledge/                   # Created at runtime
│   ├── core_memory.md
│   ├── index.json
│   ├── access_log.json
│   ├── finetune_queue.jsonl
│   ├── fundamentals/
│   ├── profession/
│   ├── projects/
│   └── personal/
│
└── tests/
    ├── test_agent.py
    ├── test_knowledge.py
    └── test_verifier.py
```

---

## CONFIG FILE (config.yaml)

```yaml
# ═══════════════════════════════════════════════
# DUAL MODEL CONFIGURATION
# Model A (Claude Sonnet) = Brain (reasoning, learning, verification)
# Model B (Qwen 7B local) = Apprentice (lookups, simple tasks, fine-tune target)
# ═══════════════════════════════════════════════

# Model A — Claude Sonnet API (The Brain)
model_a:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
  api_key_env: "ANTHROPIC_API_KEY"
  max_tokens: 2000
  temperature: 0.3
  # When to use Model A:
  tasks:
    - "task_analysis"          # Analyzing what knowledge is needed
    - "learning"               # Creating notes from sources
    - "complex_solving"        # Tasks requiring deep reasoning
    - "verification"           # Self-checking answers
    - "note_creation"          # Writing structured notes

# Model B — Qwen 2.5 7B via Ollama (The Apprentice)
model_b:
  provider: "ollama"
  model: "qwen2.5:7b"          # Base model before fine-tuning
  base_url: "http://localhost:11434"
  max_tokens: 2000
  temperature: 0.3
  # When to use Model B:
  tasks:
    - "simple_lookup"          # Answer exists verbatim in loaded notes
    - "keyword_extraction"     # Extract keywords from text
    - "note_search"            # Find relevant notes by query
    - "quick_answer"           # Simple factual Q&A from notes
    - "classification"         # Categorize tasks, topics

# Smart Router — decides which model handles each subtask
router:
  # Complexity threshold: tasks scoring above this → Model A, below → Model B
  complexity_threshold: 0.6
  
  # If Model A (API) is unavailable → fallback to Model B for everything
  fallback_to_local: true
  
  # Track API costs and switch to local when budget exceeded
  daily_api_budget_usd: 5.0
  
  # After fine-tuning, gradually shift more tasks to Model B
  auto_shift_after_finetune: true
  shift_schedule:
    v0: { model_a_pct: 90, model_b_pct: 10 }    # Before fine-tuning
    v1: { model_a_pct: 60, model_b_pct: 40 }    # After first fine-tune
    v2: { model_a_pct: 40, model_b_pct: 60 }    # Second fine-tune
    v3: { model_a_pct: 20, model_b_pct: 80 }    # Third fine-tune
    v5: { model_a_pct: 10, model_b_pct: 90 }    # Mature agent

# Fine-tuning config for Model B
finetune:
  base_model: "qwen2.5:7b"
  output_prefix: "my-agent"       # Creates: my-agent-v1, my-agent-v2, etc.
  training:
    epochs: 3
    learning_rate: 2e-5
    batch_size: 4
    lora_rank: 16
    lora_alpha: 32
    max_seq_length: 2048
  validation:
    split: 0.1
    min_examples: 50
    eval_before_switch: true       # Test new model before activating

# Knowledge Base
knowledge:
  base_dir: "./knowledge"
  core_memory_max_tokens: 4000
  auto_promote_threshold: 10
  finetune_min_examples: 50
  note_max_tokens: 1500

# Verification (always uses Model A for reliability)
verification:
  enabled: true
  min_confidence: 70
  require_sources: true
  always_use_model_a: true         # Verification = critical, use best model

# Search
search:
  method: "keyword"                # or "vector" (requires ChromaDB)
  fuzzy_threshold: 0.6

# Server
server:
  host: "0.0.0.0"
  port: 8000
```

### llm.py — Dual Model Provider (implement this)

```python
# backend/llm.py — Smart router between Claude and Qwen

import httpx
import json
from enum import Enum

class TaskType(Enum):
    TASK_ANALYSIS = "task_analysis"
    LEARNING = "learning"
    COMPLEX_SOLVING = "complex_solving"
    VERIFICATION = "verification"
    NOTE_CREATION = "note_creation"
    SIMPLE_LOOKUP = "simple_lookup"
    KEYWORD_EXTRACTION = "keyword_extraction"
    NOTE_SEARCH = "note_search"
    QUICK_ANSWER = "quick_answer"
    CLASSIFICATION = "classification"

MODEL_A_TASKS = {
    TaskType.TASK_ANALYSIS, TaskType.LEARNING,
    TaskType.COMPLEX_SOLVING, TaskType.VERIFICATION,
    TaskType.NOTE_CREATION,
}

MODEL_B_TASKS = {
    TaskType.SIMPLE_LOOKUP, TaskType.KEYWORD_EXTRACTION,
    TaskType.NOTE_SEARCH, TaskType.QUICK_ANSWER,
    TaskType.CLASSIFICATION,
}

class DualModelRouter:
    def __init__(self, config: dict):
        self.config = config
        self.api_calls_today = 0
        self.api_cost_today = 0.0

    async def call(self, task_type: TaskType, system: str, user_msg: str) -> str:
        """Route to correct model based on task type."""
        
        use_model_a = (
            task_type in MODEL_A_TASKS
            and self.api_cost_today < self.config["router"]["daily_api_budget_usd"]
            and self._api_available()
        )

        if use_model_a:
            return await self._call_claude(system, user_msg)
        else:
            return await self._call_ollama(system, user_msg)

    async def _call_claude(self, system: str, user_msg: str) -> str:
        """Call Anthropic Claude API."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.environ[self.config["model_a"]["api_key_env"]],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.config["model_a"]["model"],
                    "max_tokens": self.config["model_a"]["max_tokens"],
                    "system": system,
                    "messages": [{"role": "user", "content": user_msg}],
                },
                timeout=60,
            )
            data = resp.json()
            self.api_calls_today += 1
            # Estimate cost (~$3/M input, $15/M output for Sonnet)
            self.api_cost_today += 0.01  # rough per-call estimate
            return data["content"][0]["text"]

    async def _call_ollama(self, system: str, user_msg: str) -> str:
        """Call local Qwen via Ollama."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.config['model_b']['base_url']}/api/chat",
                json={
                    "model": self.config["model_b"]["model"],
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": False,
                },
                timeout=120,
            )
            data = resp.json()
            return data["message"]["content"]

    def _api_available(self) -> bool:
        """Check if Claude API is reachable."""
        try:
            httpx.head("https://api.anthropic.com", timeout=3)
            return True
        except:
            return False

    def update_model_b_after_finetune(self, new_model_name: str):
        """Switch Model B to fine-tuned version."""
        self.config["model_b"]["model"] = new_model_name
```

---

## API ENDPOINTS

```
POST /api/chat              # Main chat endpoint (streaming)
     Body: { "message": "...", "project": "optional_project_name" }
     Response: SSE stream with answer + verification

GET  /api/knowledge          # List all topics
GET  /api/knowledge/{topic}  # Get specific note
POST /api/knowledge/learn    # Force learn a topic
     Body: { "topic": "RS-485", "depth": "deep|quick" }
DELETE /api/knowledge/{topic} # Delete a note

GET  /api/core-memory        # View core memory
POST /api/core-memory        # Add fact to core memory
     Body: { "fact": "...", "source": "..." }
DELETE /api/core-memory      # Remove fact from core memory
     Body: { "search_text": "..." }

GET  /api/projects           # List projects
POST /api/projects           # Create project
GET  /api/projects/{name}    # Get project context

GET  /api/finetune/status    # Fine-tune queue stats
GET  /api/finetune/export    # Export JSONL for fine-tuning

GET  /api/status             # Agent status: topics count, core memory size, etc.
```

---

## CRITICAL IMPLEMENTATION RULES

### Anti-Hallucination Rules
1. The agent MUST NOT answer from general knowledge when notes exist on the topic
2. If notes are loaded, the system prompt MUST include: "Answer ONLY based on the provided notes. If information is not in the notes, say 'I don't have information about this in my notes. Should I learn about it?'"
3. Every claim in the answer should be traceable to a specific note
4. The verification step is NOT optional — always run it
5. If confidence < 70%, prefix answer with warning

### Note Quality Rules
1. Notes must be concise — only key facts, no filler text
2. Every note must have: topic, keywords, source, confidence level
3. Notes should link to related topics using [[Topic Name]] syntax
4. When agent discovers an error in a note → update immediately
5. Date-stamp every update

### Core Memory Rules
1. Core memory should stay under 4000 tokens
2. Only verified, critical facts go into core memory
3. User can always override — if they say "remember", agent remembers
4. Core memory is loaded FIRST, before any task analysis
5. Never store sensitive data (passwords, keys) in core memory

### Project Mode Rules
1. When in project mode, project notes are loaded alongside topic notes
2. Decisions should include rationale (why X, not Y)
3. Issues should include both problem and solution
4. On project end, summarize key learnings and offer to promote to core

---

## FRONTEND UI REQUIREMENTS

### Main Chat Area
- Clean chat interface, messages from user and agent
- Agent messages show confidence badge (green >90%, yellow 70-90%, red <70%)
- Show "learning..." indicator when agent is studying new topic
- Show "verifying..." indicator during self-check
- Expandable section on each answer showing: sources used, verification details

### Knowledge Sidebar
- Tree view of knowledge base (by category)
- Each topic shows: name, access count, last accessed
- Core memory section at top (always visible, editable)
- Search bar for knowledge base
- Button to force-learn new topic

### Status Bar (bottom)
- Total topics in memory
- Core memory usage (tokens/max)
- Fine-tune queue count
- Current project (if any)

---

## GETTING STARTED (for Claude Code)

1. Create the project structure
2. Implement `knowledge_manager.py` first — this is the foundation
3. Implement `core_memory.py` — three levels of memory
4. Implement `llm.py` — provider abstraction
5. Implement `agent.py` — the main loop (all 7 steps)
6. Implement `verifier.py` — self-checking
7. Build FastAPI endpoints in `main.py`
8. Build React frontend
9. Write tests
10. Create README with setup instructions

Start with a working CLI version before building the web UI.
Test the agent loop with this scenario:
- Task: "Connect Arduino Uno to RS-485 via MAX485"
- Agent should: analyze → find no notes → learn RS-485, Arduino, MAX485 → solve → verify
- Second run with same task: agent should find notes and skip learning

---

## EXAMPLE SESSION

```
User: Connect a temperature sensor DS18B20 to Arduino Uno

Agent thinking:
  1. Load core memory ✓
  2. Analyze: need [DS18B20, Arduino Uno, OneWire protocol]
  3. Check knowledge:
     - DS18B20: NOT FOUND → learning...
       → searched web → read datasheet → created note → saved
     - Arduino Uno: FOUND (accessed 5 times) → loaded
     - OneWire: NOT FOUND → learning...
       → searched web → created note → saved
  4. Solving from notes...
  5. Verifying... confidence: 94%
  6. Logging access, saving Q&A pair
  7. Cleanup: unloaded notes from context

Agent response:
  [confidence: 94%]
  Here's how to connect DS18B20 to Arduino Uno...
  (detailed answer with wiring, code, notes from knowledge base)

  📚 Learned 2 new topics: DS18B20, OneWire Protocol
  📂 Total topics in memory: 5
```
