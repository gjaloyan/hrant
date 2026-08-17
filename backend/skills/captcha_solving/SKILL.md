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

## 1. Capture the challenge — cropped

Screenshot the page, then crop to the challenge image alone and save it
to disk. Do not pass a full-page screenshot to the reader: the model was
trained on isolated challenges, and surrounding chrome measurably
degrades it.

Find the image element's box first (the DOM gives it to you — read the
`<img>` geometry through the browser rather than guessing coordinates),
then crop to exactly that box.

## 2. Establish the character count BEFORE you read

This is the step that is easy to skip and expensive to skip.

Reload the challenge two or three times, saving each image. Look at
them. Nearly every generator emits a **fixed** number of characters, and
knowing that number turns a coin-flip into a constraint:

- If every sample has 5 characters, the answer has 5 characters.
- A reading with 4 or 6 is wrong on its face — no need to submit it.

A measured failure this comes from: an agent submitted a 4-character
code to a challenge whose every sample had 5. The site rejected it and
returned to the search page, and the turn concluded the case card was
unreachable. The reading was wrong in a way one look at a second sample
would have caught.

Note anything else the samples agree on — are the glyphs only uppercase,
only digits, a mix? It narrows what a plausible answer looks like.

## 3. Read it

```
read_captcha(path="/path/to/challenge.png", expected_length=5)
```

Always pass `expected_length` once step 2 has established it.

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

On rejection, in this order:

1. **Try the next candidate.** Do this before anything else. Some glyph
   pairs — `O`/`0`/`Q`, `I`/`1`/`L`, `5`/`S`, `2`/`Z` — are genuinely
   the same shape once a font distorts them. Magnifying does not settle
   it; the site is the only thing that knows. Working down two or three
   candidates is normal and cheap.
2. **Reload for a fresh challenge** and go back to step 1. A different
   image may simply be easier — some renders are unreadable by anything.
3. After several fresh challenges have all failed, stop and report:
   how many attempts, what was read each time, and what the page did.
   That is a real finding, not a failure to deliver.

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
