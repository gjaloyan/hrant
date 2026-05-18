---
name: video_overlay_removal
description: Remove a static logo / watermark / overlay from a short Telegram video using ffmpeg's delogo filter and the multimodal LLM for coordinate detection.
triggers: ["logo", "watermark", "delogo", "overlay", "логотип", "лого", "ватермарк", "водяной знак", "убери лого", "вырез лого", "вырежи лого", "cut logo", "remove watermark", "remove overlay"]
when_to_use: |
  The user sent a video AND asks to remove a STATIC visual overlay
  (logo, watermark, channel badge, app icon, timestamp, mute/play
  button, subtitle burn-in). The overlay must stay roughly in the
  same pixel region across the clip.

  Don't reach for this when the user wants:
  - background replacement / inpainting
  - motion-tracked watermarks
  - audio modifications
  - de-blurring a region (delogo MAKES blur, doesn't remove it)
  In any of those cases tell the user plainly that delogo isn't the
  right tool.
---

# Video overlay removal — step-by-step

Hrant has a specialised vision tool — `analyze_image` — that lets
you ask the multimodal LLM a question about a JPEG. Use it to read
overlay coordinates straight off a sample frame instead of writing
OpenCV classifiers by hand. That was the failure mode of the
ad-hoc attempts: an entire iteration budget burnt on pixel arrays
and never reaching the actual render. Don't repeat it.

## Step 1 — Locate the input file

The Telegram bridge mirrors every incoming video under
`~/.hrant/data/workspace/inbox/`. Find the latest:

```bash
ls -1t ~/.hrant/data/workspace/inbox/telegram_video_*.mp4 2>/dev/null | head -3
```

If retention swept the mirror, hit the raw attachment store:

```bash
ls -1t ~/.hrant/data/knowledge/attachments/*.bin 2>/dev/null | head -5
```

Each `.bin` is keyed by sha256; the meta is in
`~/.hrant/data/knowledge/attachments/index.json` (filter
`"kind": "video"`).

## Step 2 — Probe metadata

You need width, height, and duration BEFORE you compose the filter
chain. ffmpeg's `delogo` fails fast with "Logo area is outside of
the frame" if any box exceeds the frame bounds.

```bash
ffprobe -v error \
  -show_entries stream=width,height,r_frame_rate:format=duration \
  -of default=nw=1 INPUT.mp4
```

## Step 3 — Sample frames for the vision tool

Pull two or three evenly-spaced JPEGs at SOURCE resolution (no
scale) so the coordinates you read are directly usable. The
`video_processor` helpers already exist:

```python
from pathlib import Path
from backend.tools.video_processor import _extract_frames, _probe_duration
src = Path("/home/hrant/.hrant/data/workspace/inbox/telegram_video_...mp4")
dur = _probe_duration(src)
out = Path("/tmp/overlay_frames"); out.mkdir(exist_ok=True)
frames = _extract_frames(src, out, count=3, duration=dur)
print(frames)
```

For each `frames[i]`, save it as a real attachment so `analyze_image`
can reference it by sha256:

```python
from backend.attachments import ATTACHMENTS
shas = []
for fp in frames:
    rec = ATTACHMENTS.save(fp.read_bytes(), "image/jpeg",
                           filename=fp.name, kind="image")
    shas.append(rec.sha256)
print(shas)
```

## Step 4 — Ask the multimodal LLM where the overlay is

Call `analyze_image` per sample frame with a question that demands
strict coordinates. Example question text:

> Where is the {logo/watermark/play button} in this frame? Return
> only `x=<int> y=<int> w=<int> h=<int>` in image pixels (top-left
> origin) plus one short sentence describing what you see at that
> region. If no overlay is visible, reply exactly `none`.

Validate each answer:
- `x >= 0` and `y >= 0`
- `x + w < width` and `y + h < height` (strict `<`)
- Compare across the 2-3 sample frames; if the coords drift more
  than ~10 pixels between frames the overlay is moving and `delogo`
  won't help — tell the user.

Pad the chosen box by 5-10 pixels on each side to catch
anti-aliased edges; the blur looks cleaner that way.

## Step 5 — Render with delogo

```bash
ffmpeg -y -i INPUT.mp4 \
  -vf "delogo=x=X1:y=Y1:w=W1:h=H1:show=0,fps=30,format=yuv420p" \
  -c:v libx264 -preset medium -crf 23 \
  -c:a copy -movflags +faststart \
  /home/hrant/.hrant/data/workspace/outbox/clip_no_logo.mp4
```

Chain a second `delogo=...` if there's more than one overlay.

Hard-codes that work across most Telegram videos:
- `fps=30` smooths variable rates and keeps the output
  Telegram-friendly
- `format=yuv420p` is required for broad TG client compatibility
- `-c:a copy` preserves audio without re-encoding (faster, no
  quality loss)
- `-movflags +faststart` puts MOOV at the front so the video
  starts playing while still downloading
- `-crf 23` is the balanced quality point

## Step 6 — Verify the output

Sample frames from the OUTPUT and ask `analyze_image` whether the
overlay is gone:

```bash
ffmpeg -y -i OUTPUT.mp4 \
  -vf "select='eq(n,0)+eq(n,300)+eq(n,900)'" \
  -vsync vfr /tmp/overlay_verify_%03d.jpg
```

Save each verify-frame as an image attachment, then for each:

> Is there still a {logo/watermark} visible in the {region you
> asked about}? Reply only `yes` or `no` plus one short reason.

If ANY verify-frame says `yes`, widen the box and re-render. If
the answer is `no` for all of them — ship it.

## Step 7 — Deliver

Hrant's bridge parses the answer for `MEDIA:<absolute_path>` lines
(`backend/channels.py::_strip_and_send_media`). Each such line
becomes a real Telegram attachment. Use an absolute path under
`~/.hrant/data/workspace/` so the safety allowlist accepts it.

Example final answer to the user:

```
Готово — логотип убран. Длительность исходника сохранена, аудио
без re-encode.

MEDIA:/home/hrant/.hrant/data/workspace/outbox/clip_no_logo.mp4
```

(Match the user's language — if they wrote in English, reply in
English.)

## Pitfalls

- **Empty Telegram caption ≠ no video.** A bare video without a
  caption arrives with `text == ""` while the file is already on
  disk. Always check the inbox before saying "I don't see a video".
- **`returncode 187`** from ffmpeg almost always means a malformed
  `delogo` argument — a box that exceeds the frame, or a typo. Re-
  probe metadata and re-check your coordinates.
- **0-byte output file** — ffmpeg created the destination but
  exited mid-encode. Delete the empty file before retry so a
  half-encoded artefact doesn't ship to the user.
- **Audio out of sync after re-encode** — usually means `-c:a copy`
  was dropped. Keep it unless the user asked for audio changes.
- **Logo still visible in a few frames** — overlay drifts slightly
  or your box is too tight. Widen by 8-12 pixels.
- **DON'T burn iterations writing OpenCV pixel classifiers to find
  the overlay.** Use `analyze_image`. The post-mortem on the
  pre-tool attempts showed all iterations spent on this exact
  failure pattern, never reaching the render step.

## Honesty rule

If Step 6 verification says the overlay is still visible, say so.
Don't ship a half-broken video and hope the user doesn't notice. A
bad delivery is worse than an honest "I couldn't get it" plus a
request for the coordinates explicitly.
