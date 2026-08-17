---
name: captcha_solving
description: Get past a "type the characters you see" image challenge that blocks a page you need. Covers capturing the challenge, reading it with read_captcha, submitting, and recovering from a rejection. Load this the moment a verification-code image stands between you and the data.
triggers: []
when_to_use: |
  A page you are driving shows a distorted-character image and will not
  proceed until the code is typed in. Symptoms: the form reloads with no
  result, an "enter the text on the picture" error appears, or you get
  bounced back to the search page after submitting.

  DON'T load this for: reading ordinary text out of a screenshot (use
  analyze_image), scanned documents (use an OCR engine), or checkbox /
  slider / "select all images with…" challenges — those are a different
  problem and this skill does not cover them.
---

# Getting past a character-image challenge

The whole job is four steps and one habit. The habit is the important
part: **a rejected code is information, not a dead end.** Reloading the
challenge is free, so the loop below is expected to run more than once,
and a turn that stops after one rejection has stopped too early.

## 1. Capture the challenge — through the browser, never around it

**Fetch the image with the browser you are driving, and nothing else.**
Not `curl`, not `fetch_url`, not `requests` — even against the exact URL
the page uses. A separate HTTP client carries a separate session, so the
server answers it with a *different* challenge. You will read that image
perfectly and submit an answer the page was never asking for.

This is the failure that cost a whole live run: the reader returned the
correct characters for a `curl`-fetched image, and the site rejected it
because the browser's session was validating against another one.

The recipe:

```
# 1. ask the DOM where the image actually is — never guess coordinates
eval (()=>{const r=document.querySelector('img[src*=captcha]')
           .getBoundingClientRect();
           return {x:r.x, y:r.y, w:r.width, h:r.height}})()

# 2. screenshot the page, then crop to EXACTLY that rect
screenshot /tmp/page_<unique>.png
# crop (x, y, x+w, y+h) -> /tmp/challenge_<unique>.png
```

Two details that have each broken a run on their own:

- **Crop to the reported rect, not a rounded-off guess.** A run that had
  `{x:575, y:125, w:200, h:60}` in hand cropped `(560,110,800,210)`
  anyway and pulled in the reload button and a strip of page chrome.
- **Write a fresh, uniquely-named file every time.** A run re-used a
  path from an earlier session and read a challenge that had expired six
  days earlier. It read cleanly, and it was hopeless. If the file
  already exists, you are about to read the wrong thing.

## 2. Observe the character count BEFORE you read

This is the step that is easy to skip and expensive to skip.

Reload the challenge two or three times, saving each image, and compare
them. What you are looking for is whether this generator emits a fixed
number of characters or a varying one — **both exist, and there is no
universal count.** Do not carry an expectation over from another site.

Then pass what you actually saw:

| what the samples showed | what to pass |
|---|---|
| every sample the same length *n* | `expected_length=n` |
| lengths differ (say 4 to 6) | `min_length=4, max_length=6` |
| you have not compared samples yet | nothing — leave them at 0 |

Never guess. A wrong length filter discards the correct reading, which
is worse than no filter.

A measured failure this comes from: an agent submitted a 4-character
code to a challenge whose every sample had 5. The site rejected it and
returned to the search page, and the turn concluded the record was
unreachable. The reading was wrong in a way one look at a second sample
would have caught.

Note anything else the samples agree on — only uppercase, only digits, a
mix, a fixed prefix? Each observation narrows what a plausible answer
looks like.

## 3. Read it

```
read_captcha(path="/path/to/challenge.png", expected_length=5)   # fixed-length generator
read_captcha(path="/path/to/challenge.png", min_length=4, max_length=6)  # varying
read_captcha(path="/path/to/challenge.png")                      # length not yet observed
```

What comes back:

- `best` — the top reading.
- `agreement` — whether two independent passes produced the same string.
  **This is your confidence signal.** `true` means submit `best` and
  expect it to work. `false` means the passes disagreed and the
  candidate list is where the answer probably is.
- `candidates` — ordered alternatives.

## 4. Submit, and use the candidates on rejection

Type `best`, submit, then **check what actually happened** — did the
protected content appear, or did the page bounce you back? Do not assume
success because the form accepted the keystrokes.

On rejection, first answer one question: **did the image change?**
Many sites rotate the challenge on every failed attempt, which quietly
voids everything you read from the previous one.

```
eval (()=>document.querySelector('img[src*=captcha]').src)()
```

- **Image unchanged → try the next candidate.** Some glyph pairs —
  `O`/`0`/`Q`, `I`/`1`/`L`, `5`/`S`, `2`/`Z` — are genuinely the same
  shape once a font distorts them. Magnifying does not settle it; the
  site is the only thing that knows. Working down two or three
  candidates is normal and cheap.
- **Image changed → the candidate list is dead.** Every entry describes
  a challenge that no longer exists. Go back to step 1 and capture the
  new one. Submitting a leftover candidate here is the single easiest
  way to burn attempts while looking busy.

After several freshly-captured challenges have all failed, stop and
report: how many attempts, what was read each time, and what the page
did. Suspect the mechanism rather than the reader at that point — an
expired session, a form field the page fills by script, a token tied to
the image request. That is a real finding, not a failure to deliver.

## Cross-check when it matters

`read_captcha` and `analyze_image` are independent readers. On a
challenge that keeps failing, run both on the same saved image:

- They agree → that string is very likely right, and a rejection means
  the problem lies elsewhere (an expired session, a missing form field,
  a token tied to the reload you replaced).
- They disagree → you are in the ambiguous-glyph case; work the
  candidate list.

That second reading costs one LLM call and regularly saves several
wasted submissions.

## What to report

Say what you got and what the page did with it — "read `WZSWM`, the
form accepted it and the record opened" or "four attempts, codes read
`EQX0Z` / `3VWBW` / …, each returned to the search page". Never report
the challenge as unsolvable after a single try, and never present an
unsubmitted reading as if the barrier were cleared.
