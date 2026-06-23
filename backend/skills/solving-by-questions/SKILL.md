---
name: solving-by-questions
description: Critical thinking — interrogate a task into real components, answer your own questions from vetted sources, and scale the depth to the task before building.
when_to_use: |
  Any non-trivial task, especially open-ended builds ("make a shop/app/system")
  where "what good looks like" is unclear. Load this BEFORE building so you
  solve the real problem instead of shipping a plausible-looking artifact.
---

# Solving by questions (critical thinking)

Every task starts with one question: **how do I actually solve this?** Scale the
depth to the task — do not over-think a one-liner, do not under-think a system.

## Levels — pick by size, clarity, stakes
- **L0 Reflex** — trivial/unambiguous (a calc, a reminder, a one-line fix). Just
  act. At most one framing thought.
- **L1 Frame** — simple, well-defined. Ask: *what exactly is asked? simplest
  correct solution? what would make it wrong?* Then act.
- **L2 Structure** — moderate, multi-step. Ask: *what are the parts? the
  approach? inputs/data? edge cases? what does "done" look like?* `set_plan`,
  then execute.
- **L3 Interrogate & scope** — big / open-ended build. **Building an app / shop /
  site / system is ALWAYS at least L3.** A given detail (a port, a name, a colour)
  does NOT make it scoped — the scope is *which components are real*, and that is
  exactly what you must not assume. Ask: *what IS this thing? what are its REAL
  components? what does a functional version need (NOT a demo)? data model? flows?
  MVP vs full? what is unknown?* Then call `frame_problem` to record the component
  map + a proposed scope, confirm scope with the owner via `ask_user`, and build
  only the confirmed scope.
- **L4 Project** — too big for one session. Materialize a project: goals, a
  short spec, a plan, decomposed tasks; persist state in files (`create_tracker`
  + workspace docs). Then build it step by step by **delegating each task to a
  fresh `builder` subagent** (and `researcher` for fact-finding, `reviewer` for a
  second opinion) — subagent-driven, so no single context carries the whole
  project and nothing gets forgotten. Review each result before the next.

## Use subagents often — and let them DO, not just read
A subagent is a fresh, focused context. Use `delegate(role, task)` whenever a
piece of work is self-contained — to keep YOUR context clean and avoid
forgetting. Roles: `researcher` (web + citations), `builder` (implements a
focused task end to end — writes files, runs code, starts processes, verifies
it), `reviewer` (critiques before you commit). Give the subagent a
self-contained task (it cannot see your conversation). Don't delegate trivial
one-liners — delegation has overhead — but on anything multi-part, delegating
the separable pieces beats doing it all in one window.

## Answering discipline — how to trust your own answers
A question is only as good as its answer. The web is necessary but noisy.
1. **Own knowledge first** — `search_knowledge`, your facts/trajectories. You may
   already hold a vetted answer, and it carries the owner's context.
2. **Source hierarchy** — primary/authoritative (official docs, specs, source,
   recognized authorities) > reputable media > forums/UGC > SEO/content-farms.
3. **Triangulate** — confirm load-bearing claims across 2-3 independent quality
   sources. Consensus kills outliers and fakes.
4. **Critique with reason** — is it plausible? consistent? matches first
   principles? You judge; you do not copy.
5. **Verify by running** (where checkable) — code, an API call, a test. A fact
   from experiment beats any source.
6. **Cache** — `save_knowledge` what you proved, so next time own-knowledge is
   first.
7. **Escalate** — sources conflict or low confidence? `ask_user`, do not ground
   on something shaky.

Scale rigor to stakes: trivia → one good source; a decision the whole build
rests on → triangulate and verify.

## The trap this prevents
Asked to "build an online shop," do NOT jump to a pretty page. Interrogate first
(catalog, cart, checkout, payments, accounts, inventory, orders store, admin,
auth), `frame_problem` the map, confirm scope, then build the real thing.
