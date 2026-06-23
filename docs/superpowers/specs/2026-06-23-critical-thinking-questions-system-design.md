# Critical-Thinking (Questions) System — Design

**Goal:** Give the agent a critical-thinking faculty so it solves the *actual*
problem instead of producing a plausible-looking artifact. The trigger case: asked
to "build an online shop," the agent shipped a decorative single page — it never
asked itself *what a functional online shop actually is and what it's made of*. The
fix is a universal, leveled self-questioning mechanism with a disciplined way of
answering those questions.

**Essence:** This is one thing — **critical thinking** — operationalized as: ask the
right questions to figure out how to solve a task, scale the depth to the task, and
answer each question from vetted sources (your own knowledge first), not from a
single guess or a single blog.

---

## 1. Core concept

Every task begins with a question to oneself: **"how do I actually solve this?"** The
*depth* of that questioning scales with the task. It is not on/off — it is a gradient
(levels). It applies everywhere, from a one-line fix to a multi-week system; only the
depth changes. The same faculty also governs how answers are trusted (the answering
discipline) and how very large work is carried (a persisted project).

This subsumes and connects the two principles already in the soul — *research before
you build* and *big work does not fit in your head* — under one idea: think
critically.

## 2. The levels

The agent picks a level by judging the task's size, clarity, and stakes (model
judgment, guided by the skill).

- **L0 — Reflex.** Trivial, unambiguous (greeting, a calculation, a reminder, a
  one-line fix). No explicit questioning; just act. At most one framing thought.
  Over-questioning here is waste.
- **L1 — Frame.** Simple and well-defined ("fix this function", "summarize this").
  1–2 quick self-questions: *what exactly is asked? what's the simplest correct
  solution? what would make it wrong?* Then act.
- **L2 — Structure.** Moderate, multi-step ("add feature X", "analyze Y").
  Structuring questions: *what are the parts? what's the approach? what inputs/data?
  what edge cases? what does "done" look like?* Form a mini-plan (`set_plan`), execute.
- **L3 — Interrogate & scope.** Big / open-ended where "good" is unclear ("build a
  shop/app/system"). Full method: *what IS this thing? what are its REAL components?
  what does a functional version require (not a demo)? data model? flows? MVP vs full?
  what's unknown?* → build a **component map** (the tool) → propose a realistic scope →
  **confirm with the owner via `ask_user`** → decompose into steps → build.
- **L4 — Project.** When the confirmed scope will not fit one working session (a real
  multi-component system, doesn't fit the context window): materialize a durable
  **project** — goals, a short spec, a plan, decomposed tasks — and execute
  incrementally across sessions, with state living in files (a tracker + workspace
  docs), loading only the slice each step needs. This is the agent running its own
  brainstorm → plan → execute pipeline, the same shape as the superpowers flow.

## 3. The answering discipline (how questions get answered reliably)

A self-question is only as good as its answer. The web is necessary but noisy —
grounding on a confident-but-wrong source is just sourced hallucination. So each
answer is earned through layers, scaled to stakes:

1. **Own knowledge first.** Check the agent's own memory, notes, and trajectories
   (`search_knowledge`, facts) before reaching outward — it may already hold a vetted
   answer, and it carries the owner's context.
2. **Source hierarchy.** Prefer primary/authoritative sources (official docs, specs,
   source code, recognized authorities) over reputable media over forums/UGC over
   SEO/content-farms.
3. **Triangulation.** Confirm load-bearing claims across 2–3 *independent* quality
   sources. Consensus beats a single voice and kills outliers/fakes.
4. **Critique with reason.** The model judges what it found — plausible? internally
   consistent? consistent with first principles? This is the model's job: judgment
   over retrieved data, not copying.
5. **Empirical verification (where possible).** For technical/checkable claims, run it
   — code, an API call, a test. A fact from experiment outranks any source. (Only
   applies to verifiable questions, not to taste questions like "what is good UX.")
6. **Cache the vetted answer.** Save what was verified back to the knowledge base
   (`save_knowledge`) so next time own-knowledge-first already has it.
7. **Escalate on conflict / low confidence.** When sources disagree or confidence is
   low, ask the owner (`ask_user`) rather than ground on something shaky.

**Calibration:** scale rigor to stakes. Trivia → one good source. A decision the whole
build rests on → triangulate and verify. Do not triangulate erunda — latency is real.

**Honest limits:** source authority is heuristic, not certain. Some questions have no
factual answer (subjective/novel) — then reason from principles or ask, don't dress a
blog as truth. Verification-by-running only covers checkable claims.

## 4. The three layers (form)

The user's framing: soul = *know* (disposition/when), skill = *know how* (method),
tool = *structure*.

### 4.1 Soul — disposition (English, in `How You Act`)
A tight critical-thinking principle that fires the mechanism and points at the method.
It must not duplicate the existing *research-first* / *decompose* lines but tie them
together. Proposed text:

> **Think before you build — question, then verify.** Every task starts with a
> question to yourself: how do I actually solve this? Scale the questioning to the
> task — a one-liner needs a thought; a system needs interrogation. For anything
> non-trivial, load `solving-by-questions` and use it. Never build something big
> without first interrogating it into its real components (what does a *functional*
> version need, not a demo?), mapping them, and confirming scope with your owner.
> Answer your own questions from vetted ground, not a guess: your own knowledge and
> memory first, then authoritative sources; triangulate the load-bearing claims;
> verify by running what you can; cache what you proved; and when sources conflict or
> you're unsure, ask rather than assume. When the work won't fit one sitting, make it
> a project — goals, plan, tasks in files — and build it step by step.

### 4.2 Skill — method (`solving-by-questions`)
A loadable skill holding the *how*:
- The leveled question templates (L0–L4) and how to pick a level.
- The L3 interrogation set + how to build a component map + how to propose scope.
- The L4 project method (goals → spec → plan → tasks; state in files; incremental).
- The full **answering discipline** (section 3).
The soul tells the agent to load this for non-trivial tasks.

### 4.3 Tool — structure (`frame_problem`)
One new tool for the heavy case (L3/L4). It records and persists the structured frame:
- **component map** — each component with: needed-in-MVP vs later, source, confidence;
- **proposed scope** (what to build now vs defer) + **open questions**;
- writes a durable artifact (survives the context window);
- drives the **scope confirmation** through `ask_user`; the confirmed scope becomes the
  spec the build follows. For L4, the confirmed scope seeds a `create_tracker` project.

## 5. Reuse (do not reinvent)
- `ask_user` → scope confirmation + escalation on conflict.
- `set_plan` / `update_plan` → L2 mini-plan, L3 step decomposition.
- `search_knowledge` / `save_knowledge` → own-knowledge-first + caching vetted answers.
- `web_search` / `fetch_url` / `agent_browser` → authoritative research + triangulation.
- `create_tracker` → L4 persisted multi-session project (goals + steps).
- `run_python` / `terminal_exec` → empirical verification.

So the only genuinely new code is the `frame_problem` tool. The rest is a soul edit, a
markdown skill, and wiring existing pieces.

## 6. Non-goals (YAGNI)
- No automatic source-credibility scoring engine — authority/triangulation are method
  guidance in the skill, judged by the model, not a scored subsystem.
- No new persistence layer — reuse the tracker + workspace files.
- No forced questioning on L0/L1 trivia — calibration must keep small tasks fast.

## 7. Testing
- `frame_problem` tool: unit tests for artifact shape (component map, scope, sources,
  confidence) and that it routes scope to `ask_user`.
- Soul/skill: a reachability guard (skill loads; soul references it) — behavior is
  validated by live behavioral probes (the shop case re-run), not unit tests.
- Calibration: a probe that a trivial task does NOT trigger L3 interrogation.
