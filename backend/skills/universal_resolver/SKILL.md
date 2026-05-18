---
name: universal_resolver
description: Fallback workflow for UNKNOWN file types or UNKNOWN tasks. Load this when the AVAILABLE SKILLS catalog has no entry that matches, and the task is non-trivial (not a chat reply, not a directive with a known set_setting / grant_telegram_access / etc. target).
triggers: []
when_to_use: |
  Read this skill before saying "I don't know how to do this".

  Activate when ALL of the following hold:
  - the user's request is non-trivial (not chat / not a recall question)
  - no other skill in AVAILABLE SKILLS has triggered or obviously fits
  - the task involves:
      * a file format you don't have a documented reader for
        (CorelDRAW .cdr, AutoCAD .dwg, Sketch .sketch, database
        dump .sqlite/.dump, special archive .rar/.7z, etc.),
      * a piece of software you've never been asked about,
      * a conversion / extraction / editing / automation task
        whose exact pipeline you can't recall from memory,
      * or any other "this is outside my catalogue" feeling

  DON'T read this for: chat, recall, a known multi-channel grant,
  setting changes, web search of a fact. Those have their own
  paths and don't need a fallback.
---

# Universal Unknown Resolver

This is the path you walk when the catalogue is silent. It exists
so the answer "I can't do this" stops being the first reply.

## Rule of thumb for the first sentence of your final reply

NEVER open with:
- "I don't know how to do this."
- "I have no tool for that."
- "This isn't supported."

ALWAYS instead state the plan you intend to follow, in one short
sentence, then walk the steps below. Even if you end up
unsuccessful, the user should see what was tried.

A template you can adapt to the user's language:

> Эта задача не зарегистрирована в моих skills. Я разберу запрос,
> поищу проверенные методы, найду подходящие инструменты, безопасно
> протестирую workflow, выполню если возможно, и сохраню рабочий
> результат как переиспользуемый skill.

Or in English:

> This task isn't in my skill registry. I'll analyse the request,
> research reliable methods, find suitable tools or libraries, test
> the workflow safely, complete the task if possible, and save the
> successful workflow as a reusable skill.

Don't QUOTE the template literally — adapt the language and tone
to the user. The point is committing in plain words to the
workflow before walking it.

## Workflow — seven phases, in order

### 1. Understand the request

Re-read the user's message slowly. What is the **goal**, not just
the surface verb? Then classify into one or more dimensions:

- File-related task (read / convert / edit / extract / repair)
- Software interop (open / drive / scrape / parse)
- Conversion between formats
- Data extraction (rows, fields, text from binary, OCR from image)
- Media processing (image / video / audio transformation)
- Automation (run a sequence of operations on N inputs)
- Coding (write / patch / generate / explain code)
- Design / creative (less common — be explicit if you go here)

If any attachments are in play, note their `sha256` and what
`AttachmentStore` already knows about them (`kind`, `mime_type`,
`filename`, existing `transcript`, `frame_shas`).

### 2. Check what you already have

Don't reinvent. Walk the inventory:

- `list_skills()` — full catalogue with descriptions + triggers.
  Even if no trigger fired, a description might match the task
  shape; `load_skill(name)` reads the body.
- Recent successful turns for this speaker: `search_knowledge` on
  the task domain (the conversation buffer may carry a known good
  recipe).
- The full tool registry: read each tool's description before
  reaching for `terminal_exec` / `run_python` — sometimes a
  dedicated tool like `analyze_image` or `read_file` covers what
  you were about to write code for.

If you find a near-match skill, ADAPT it (call `load_skill`, then
write a one-paragraph plan describing how this task differs and
what you'll change). Don't blindly follow a skill that doesn't
quite fit.

### 3. Identify what's missing

State explicitly (to yourself, in the progress trace via
`agent.progress` if available) what you still need to acquire:

- a file reader (parser / decoder / extractor)
- a converter (format A → format B)
- a Python library (pillow, pandas, pypdf, ffmpeg-python, …)
- a CLI tool (ffmpeg, libreoffice, imagemagick, qpdf, …)
- an API endpoint (only if the data lives remotely)
- documentation (you don't actually know how the format works)
- a processing method (algorithmic gap — research the algorithm)
- a task-specific workflow (chain of tools you've never run
  together)

Being explicit here makes Step 4 cheap — you search for the
specific gap, not the entire problem.

### 4. Research (and only research)

Use `web_search` followed by `fetch_url` on the most-promising
result. Prefer sources in this priority order:

1. Official documentation: language stdlib docs, the project's own
   GitHub README / wiki, vendor docs.
2. Package registries: pypi.org, crates.io, npmjs.com — the
   project's page tells you the canonical install command and
   maintainer.
3. Established Q&A: StackOverflow accepted answers, GitHub issues
   on the official repo.
4. Reputable blogs: only when the official source explains
   nothing and the blog clearly cites versions / sources.

NEVER follow a "magic one-liner" from a random blog without
checking it against the official docs first. NEVER paste curl
| bash from a non-vendor source.

For Russian-language tasks: it's fine to research in English even
if the user wrote in Russian; the answer will come back in the
user's language anyway.

### 5. Choose tools, safely

Walk the priority ladder. Stop at the first one that solves the
task; only descend on real failure.

1. Built-in tools and skills already in the registry.
2. Already-installed binaries: check with `which <name>` or
   `apt list --installed 2>/dev/null | grep <name>`. ffmpeg,
   libreoffice, imagemagick, qpdf, etc. are likely present.
3. Already-installed Python libraries: try `python3 -c "import X"`
   before pip-installing — many libraries are pre-loaded.
4. NEW installs — owner approval is REQUIRED. Go through
   `propose_install(packages, manager, reason)`. The tool writes
   a pending request and DMs the owner `[👀 Show] [✅ Approve]
   [❌ Reject]` inline buttons; only on Approve does the install
   command actually run.
   - Supported managers: `pip` (default), `pipx`.
   - `apt` / `npm -g` are NOT supported by the gate — sudo /
     root decisions sit outside this layer. Ask the owner to
     run them manually if absolutely needed.
   - Calling `pip install` / `pipx inject` / `apt install` /
     `npm install` / `cargo install` etc. through `terminal_exec`
     will be REFUSED with an error pointing back here.
   - After `propose_install`, STOP and wait. Don't assume the
     package is importable in the same turn — the install runs
     in the next turn after the owner approves.

The reasoning isn't paranoia about your own choices — it's that
any auto-install is a supply-chain decision the owner needs to
sign for. A wrong install creeps into the agent's runtime
permanently.

### 6. Test safely

Before you produce a deliverable:

- Run the workflow on a **copy** of the input, never the original
  source file. Copy with `cp` or Python `shutil.copy` into
  `/tmp/<turn>/`.
- For unknown executables / archives / scripts: **do not** run
  them directly. Open with the chosen tool (libreoffice headless,
  unrar -l, etc.) without executing embedded macros / payloads.
- Watch for non-zero exit codes. ffmpeg `returncode 187`,
  imagemagick policy.xml refusals, libreoffice timeout — surface
  the actual error, don't paper over it.
- Verify the output:
  - For media: re-probe with `ffprobe` / `Pillow.open` /
    `pdfinfo` / `analyze_image` (vision) to confirm the file
    isn't a 0-byte stub or a half-encoded artefact.
  - For converted text: spot-check first and last paragraphs.
  - For extracted data: confirm row counts / non-empty columns.

If verification fails, iterate ONCE before falling back to
"here's what I tried — please clarify". Don't ship broken output
hoping the user doesn't notice.

### 7. Solve and deliver

When the workflow holds:

- Write the result to `~/.hrant/data/workspace/outbox/` with a
  descriptive filename (so a future turn can find it).
- In the final reply, include exactly one `MEDIA:/absolute/path`
  line per deliverable file (see RULES → "Sending files back").
- Brief plain-text summary alongside: what was done, any
  caveats. Match the user's language.

### 8. Save the workflow as a reusable skill (the closing step)

ONLY when:
- the task was non-trivial (real research and / or composed tool
  sequence — not just `read_file` on a known format),
- the result is verified-good,
- the same shape of task is plausibly going to recur (logo
  removal: yes; "open this very specific spreadsheet": no).

Then call:

```
propose_skill(
  name="kebab-case-name",
  description="One sentence — what this skill does.",
  triggers="comma,separated,distinctive,keywords",
  when_to_use="When to reach for this vs related skills.",
  body="# Step-by-step Markdown. Be explicit about file paths,
        exact commands, expected outputs, and pitfalls. Future
        turns READ THIS VERBATIM."
)
```

The skill is written DISABLED. The owner sees a Telegram DM with
inline buttons `[👀 Show] [✅ Activate] [❌ Delete]`. Don't
expect it to be live this turn — that's correct, and it's what
makes the auto-create safe.

## Pitfalls (real failures from the audit logs)

- **Spending 10 iterations on "extract one frame" sub-problems.**
  When you're 6 calls deep and still don't have a clear path,
  STOP and write the status report — see RULES → "Iteration
  ceiling". Don't dump `<tool_call ...>` XML.
- **"Tools are disabled" hallucination.** They aren't. If your
  first attempt failed, try a DIFFERENT tool — pivot, don't
  surrender.
- **`pip install` without asking.** Supply-chain risk; the owner
  must sign for installs. See Step 5.
- **Verbose 5000-word research dumps in the final reply.** The
  user wanted the result, not the path. Keep the answer tight;
  trace details ride in the SSE stream / WebUI tools panel.
- **Forgetting MEDIA: when a file is the deliverable.** Without
  the line, the file stays on disk and the user thinks the bot
  is broken.
- **Marking a half-success as "done".** If verification said the
  output is suspect, SAY SO. A clear "couldn't get there, here's
  why" beats a wrong silent ship.
