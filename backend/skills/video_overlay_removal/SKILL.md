---
name: video_overlay_removal
description: Remove a static logo / watermark / overlay from a short Telegram video using ffmpeg's delogo filter.
triggers: ["logo", "watermark", "delogo", "overlay", "логотип", "лого", "ватермарк", "водяной знак", "убери лого", "вырез лого", "вырежи лого", "cut logo", "remove watermark", "remove overlay"]
when_to_use: |
  The user sent a video (mp4 / mov / webm) AND asks to remove a
  static or near-static visual overlay from it: logo, watermark,
  channel badge, app icon, timestamp, mute/play button, subtitle
  burn-in, etc. The overlay must stay roughly in the SAME PIXEL
  REGION across the clip — moving overlays need a different
  approach (frame-by-frame masking or a real inpainter).

  This is NOT for: full background replacement, deep object removal,
  motion-tracked watermarks, or audio-side modifications. Be explicit
  with the user if the requested removal needs something more
  sophisticated than `delogo` — ffmpeg's delogo creates a blurred
  patch over the region, not a clean reconstruction.
---

# Video overlay removal

## Goal
Take an incoming Telegram video, identify the bounding box of the
unwanted overlay, mask each frame with ffmpeg's `delogo` filter,
re-encode, verify, then send the cleaned file back to the user via
the MEDIA: convention.

## Where the input lives
Telegram videos land in the attachments store, mirrored at:

```
~/.hrant/data/workspace/inbox/telegram_video_<file_unique_id>.mp4
```

Find the most recent one if the user didn't quote a specific name:

```bash
ls -lt ~/.hrant/data/workspace/inbox/telegram_video_*.mp4 2>/dev/null | head -3
```

Also check the AttachmentStore index — it's authoritative when the
inbox mirror was swept by retention:

```bash
ls -lt ~/.hrant/data/knowledge/attachments/*.bin 2>/dev/null | head -3
```

Each `.bin` is the raw upload; the mime-type lives in
`~/.hrant/data/knowledge/attachments/index.json`. Filter for
`"kind": "video"`.

## Step 1 — Probe the file
You need width, height, and duration before composing the filter
chain. `delogo` fails fast with "Logo area is outside of the frame"
if any `x + w` or `y + h` exceeds the frame.

```bash
ffprobe -v error -show_entries stream=width,height,r_frame_rate:format=duration \
  -of default=nw=1 INPUT.mp4
```

Or via Python (the module is already in the engine):

```python
from backend.tools.video_processor import _probe_duration
dur = _probe_duration(Path("INPUT.mp4"))
```

## Step 2 — Sample frames to locate the overlay
Pull a small set of evenly-spaced JPEGs so vision (or you, on the
follow-up turn) can read the overlay's pixel coordinates:

```python
from pathlib import Path
from backend.tools.video_processor import _extract_frames
out = Path("/tmp/overlay_inspect")
out.mkdir(exist_ok=True)
frames = _extract_frames(Path("INPUT.mp4"), out, count=4, duration=dur)
print(frames)
```

Frames are written as JPEGs at long-edge 1280px. If the source was
LARGER (e.g. 1920×1080), remember the scale factor when reading
coordinates back. Frame-coord `(x, y)` on a 1280-scaled frame becomes
`(x * src_w / 1280, y * src_h / (1280 * src_h / src_w))` on the
original — usually you can just keep the frames at source resolution
by passing `count` only and skipping the scale step.

## Step 3 — Pick the bounding boxes
Open the JPEG (or describe it to the user) and read off `(x, y, w, h)`
for each overlay region. Hrant's response can include image
attachments; if you sent the sampled frames as attachments, the
vision-capable LLM that sees them can dictate coordinates.

Sanity checks before composing the filter:
- `x >= 0` and `y >= 0`
- `x + w < width` and `y + h < height` (use `<`, not `<=` — ffmpeg
  is strict)
- For each region, make the box ~5-10px bigger than the visible
  overlay so the blur covers anti-aliased edges.

## Step 4 — Compose the filter chain
Chain one `delogo` per overlay:

```
delogo=x=X1:y=Y1:w=W1:h=H1:show=0,
delogo=x=X2:y=Y2:w=W2:h=H2:show=0
```

`show=0` hides the green debugging rectangle. While iterating you
can flip it to `show=1` to verify boxes visually on the first frame.

## Step 5 — Render
Hard-codes that work across most Telegram videos:

```bash
ffmpeg -y -i INPUT.mp4 \
  -vf "delogo=x=X1:y=Y1:w=W1:h=H1:show=0,fps=30,format=yuv420p" \
  -c:v libx264 -preset medium -crf 23 \
  -c:a copy -movflags +faststart \
  OUTPUT.mp4
```

Notes:
- `fps=30` smooths out variable / high frame rates and keeps the
  output Telegram-friendly.
- `format=yuv420p` is needed for broad Telegram client compatibility.
- `-c:a copy` preserves audio without re-encoding (saves time and
  prevents quality loss).
- `-movflags +faststart` puts the MOOV atom at the start so the
  video starts playing while still downloading.
- `-crf 23` is a balanced quality/size point; lower (~18) for higher
  quality, higher (~28) for smaller file.

## Step 6 — Verify before sending
ALWAYS extract a couple of output frames and check the overlay is
actually gone, AND that the blur didn't smudge important content:

```bash
ffmpeg -y -i OUTPUT.mp4 \
  -vf "select='eq(n,0)+eq(n,300)+eq(n,900)'" \
  -vsync vfr /tmp/overlay_verify_%03d.jpg
```

If a frame still shows the logo: widen the box and re-render. If
the blur is obviously hiding wanted content: narrow the box.

## Step 7 — Deliver back to Telegram
Hrant's reply is parsed for `MEDIA:<absolute_path>` lines (see
[backend/channels.py](backend/channels.py) — `_strip_and_send_media`).
Each such line becomes a real attachment in the user's chat.

Place the final answer like:

```
Готово — логотип убран. Время: 12.3s.
MEDIA:/home/hrant/.hrant/data/workspace/outbox/clip_no_logo.mp4
```

(Or English text, depending on the user's language.) Use an
absolute path under `~/.hrant/data/workspace/` so the safety
allowlist accepts it. `/tmp` is also allowed for ephemeral output.

## Pitfalls
- **An empty-looking Telegram message does NOT mean no video was
  sent.** Photos / videos without captions arrive with `text == ""`
  while the file is already on disk in the attachments dir. Always
  check the inbox before telling the user "I don't see a video".
- **`returncode 187`** from ffmpeg almost always means an invalid
  filter argument: a `delogo` box that exceeds the frame, or a
  malformed expression. Re-probe metadata and re-check coordinates.
- **0-byte output file** — ffmpeg created the destination but
  exited mid-encode. Usually pairs with the previous pitfall.
  Delete the empty file before retry so a half-encoded artifact
  doesn't ship to the user.
- **Audio out of sync after re-encode** — happens when `-c:a copy`
  was dropped. Always keep it unless the user asked for an audio
  edit.
- **Logo still visible in a few frames** — overlay slightly moves,
  or the box was too tight. Widen by 8-12 pixels on each side.

## Honesty rule
If verification (Step 6) shows the result is bad — say so. Don't
ship a half-broken video and hope the user doesn't notice. A bad
delivery is worse than admitting the current attempt failed and
asking for the overlay coordinates explicitly.
