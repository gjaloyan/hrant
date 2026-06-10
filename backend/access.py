"""Unified access control + pairing system — Phase B+C.

Hermes audit caught the 2-file access problem:
  - `channels.json::allowed_users` — telegram bot filter (string IDs)
  - `roles.json::owner_speaker_ids` + `speakers.<id>.role` — semantic roles
        (owner / trusted / guest)

Adding a user required updating BOTH atomically. Agent didn't know
about the second file → wrote only to `roles.json` → bot still
blocked the user → 4 turns to fix what hermes did in 1.

This module is the single entry point. `channels.py` calls
`is_telegram_allowed(user_id, username)` instead of poking
allowed_users directly. The function:

  1. Checks role from `roles.json` (owner/trusted → allow)
  2. Falls back to legacy `channels.json::allowed_users` (allow,
     with a migration nudge)
  3. Falls back to pairing system (Phase C): create a pending
     pairing request, notify owner, tell the stranger to wait.

For mutations the agent has two tools — `grant_telegram_access`
(when the owner gives a user_id directly) and `approve_pairing`
(when an unknown user already wrote and there's a pending code).
Both update the right places atomically.

Pairing system (Phase C):
  - Unknown telegram user writes → access.py creates a pairing
    request with a short code (8 chars from an unambiguous alphabet)
    and a 1-hour TTL.
  - The bot DMs the owner: "User Lusine Sargsyan (id 1562235884)
    wants access. First message: '<snippet>'. To approve, reply
    `approve <code>` OR call the tool."
  - Stranger is told: "Your access request is pending owner
    approval."
  - On approve: the user is moved into `roles.json::trusted` AND
    the request is cleared.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from . import paths


log = logging.getLogger(__name__)


# ─── Pairing constants ──────────────────────────────────────────────


# 32-char alphabet (no 0 / O / 1 / I — confusable). Matches hermes'.
_PAIRING_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
_PAIRING_CODE_LEN = 8
_PAIRING_TTL_SECONDS = 3600           # 1 hour
_PAIRING_MAX_PENDING_PER_PLATFORM = 5


# ─── Decision shape returned by check_telegram_access ───────────────


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    role: str                          # owner / trusted / guest / unknown
    reason: str                        # short human-readable
    pairing_code: str = ""             # set only when a new request was created
    pairing_pending: bool = False      # already had an open request


# ─── Pairing request ────────────────────────────────────────────────


@dataclass
class PairingRequest:
    """One pending access request from an unknown user."""

    code: str
    platform: str                      # "telegram"
    user_id: str                       # platform-native id (string form)
    username: str                      # @handle if known, else ""
    label: str                         # full_name / display
    first_message: str                 # snippet, capped
    created_at: float
    expires_at: float
    requester_speaker_id: str          # e.g. "telegram:1562235884"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PairingRequest":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})


# ─── PairingStore ───────────────────────────────────────────────────


class PairingStore:
    """File-backed store for pending pairing requests.

    State lives at ~/.hrant/data/knowledge/pairing.json:
      { "requests": [PairingRequest dicts] }

    Atomic write via .tmp rename. Thread-safe through RLock."""

    def __init__(self, path: Optional[Path] = None):
        self._path_override = path
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        if self._path_override is not None:
            return self._path_override
        return paths.knowledge_dir() / "pairing.json"

    def _load(self) -> dict:
        try:
            p = self.path
            if not p.exists():
                return {"requests": []}
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("pairing store load failed: %s", e)
            return {"requests": []}

    def _save(self, state: dict) -> None:
        try:
            from . import paths as _paths
            _paths.write_secret_json(self.path, state)
        except Exception as e:
            log.warning("pairing store save failed: %s", e)

    def _gc(self, state: dict) -> dict:
        """Drop expired requests. Called on every read/write so a
        single user can't fill the queue with stale codes."""
        now = time.time()
        kept = [r for r in state.get("requests", []) if float(r.get("expires_at") or 0) > now]
        state["requests"] = kept
        return state

    # ─── Public API ─────────────────────────────────────────────────

    def list_pending(self, platform: str | None = None) -> list[PairingRequest]:
        with self._lock:
            state = self._gc(self._load())
            self._save(state)
            rows = [PairingRequest.from_dict(r) for r in state.get("requests", [])]
        if platform:
            rows = [r for r in rows if r.platform == platform]
        rows.sort(key=lambda r: r.created_at)
        return rows

    def get_by_user(
        self, platform: str, user_id: str,
    ) -> Optional[PairingRequest]:
        for r in self.list_pending(platform=platform):
            if str(r.user_id) == str(user_id):
                return r
        return None

    def get_by_code(self, code: str) -> Optional[PairingRequest]:
        code_norm = (code or "").strip().upper()
        if not code_norm:
            return None
        # Allow partial prefix match (≥4 chars) so the owner doesn't
        # have to type the full code if it's unique.
        candidates = [
            r for r in self.list_pending()
            if r.code.startswith(code_norm) or code_norm.startswith(r.code)
            or (len(code_norm) >= 4 and r.code.startswith(code_norm))
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            return None
        # Multiple matches — only accept the exact one.
        for c in candidates:
            if c.code == code_norm:
                return c
        return None

    def create(
        self,
        *,
        platform: str,
        user_id: str,
        username: str,
        label: str,
        first_message: str,
        ttl_seconds: int = _PAIRING_TTL_SECONDS,
    ) -> Optional[PairingRequest]:
        """Idempotent: returns the EXISTING request if this user
        already has one pending; otherwise creates a fresh one.
        Refuses (returns None) when the platform is at its max
        pending cap — a spammer can't flood the queue with codes."""
        with self._lock:
            state = self._gc(self._load())
            # Idempotency
            for r in state.get("requests", []):
                if (r.get("platform") == platform
                        and str(r.get("user_id")) == str(user_id)):
                    return PairingRequest.from_dict(r)
            same_platform = [
                r for r in state.get("requests", [])
                if r.get("platform") == platform
            ]
            if len(same_platform) >= _PAIRING_MAX_PENDING_PER_PLATFORM:
                log.warning(
                    "pairing: refusing new request — platform %r at cap %d",
                    platform, _PAIRING_MAX_PENDING_PER_PLATFORM,
                )
                return None
            code = "".join(secrets.choice(_PAIRING_ALPHABET) for _ in range(_PAIRING_CODE_LEN))
            now = time.time()
            req = PairingRequest(
                code=code,
                platform=platform,
                user_id=str(user_id),
                username=username or "",
                label=label or "",
                first_message=(first_message or "")[:300],
                created_at=now,
                expires_at=now + ttl_seconds,
                requester_speaker_id=f"{platform}:{user_id}",
            )
            state.setdefault("requests", []).append(req.to_dict())
            self._save(state)
            return req

    def consume(self, code_or_user_id: str, platform: str = "telegram") -> Optional[PairingRequest]:
        """Look up the request by code OR by user_id and remove it
        from the pending store. Used at approval time."""
        with self._lock:
            state = self._gc(self._load())
            requests = state.get("requests", [])
            target_idx = -1
            target = None
            ident = (code_or_user_id or "").strip()
            ident_upper = ident.upper()
            for i, r in enumerate(requests):
                if r.get("platform") != platform:
                    continue
                if str(r.get("user_id")) == ident or r.get("code") == ident_upper:
                    target_idx = i
                    target = r
                    break
            if target_idx >= 0:
                requests.pop(target_idx)
                state["requests"] = requests
                self._save(state)
            return PairingRequest.from_dict(target) if target else None


PAIRING_STORE = PairingStore()


# ─── Unified access checker ─────────────────────────────────────────


def is_telegram_allowed(
    user_id: int | str,
    username: str = "",
    *,
    legacy_allowed: Optional[list[str]] = None,
    first_message: str = "",
    label: str = "",
) -> AccessDecision:
    """Decide whether to admit a Telegram message into the agent.

    Resolution order:
      1. Role check via `roles.role_of("telegram:<id>")`:
         - owner / trusted → allow immediately
      2. Legacy `channels.json::allowed_users` list (numeric ID or
         username match) → allow with a migration note. New
         installs should not rely on this; the unified role API
         is the path forward.
      3. Pairing: create a pending request (idempotent) and return
         a decision telling the channel to notify the owner and
         hold the stranger.

    `legacy_allowed` is the bot's old allowed_users list; pass it in
    so we don't have to read channels.json from this module (avoids
    an import cycle).
    """
    from . import roles as _roles

    uid_str = str(user_id)
    speaker_id = f"telegram:{uid_str}"

    # Layer 1: role-based
    try:
        role = _roles.role_of(speaker_id)
    except Exception:
        role = "guest"
    if role in ("owner", "trusted"):
        return AccessDecision(
            allowed=True, role=role,
            reason=f"role={role}",
        )

    # Layer 2: legacy allowed_users (string match against id or username)
    if legacy_allowed:
        if uid_str in legacy_allowed or (username and username in legacy_allowed):
            return AccessDecision(
                allowed=True, role="trusted",
                reason="legacy allowed_users match (consider migrating to roles)",
            )

    # Layer 3: pairing — unknown user, create / fetch pending request
    existing = PAIRING_STORE.get_by_user("telegram", uid_str)
    if existing is not None:
        return AccessDecision(
            allowed=False, role="unknown",
            reason="pairing request pending owner approval",
            pairing_code=existing.code,
            pairing_pending=True,
        )
    req = PAIRING_STORE.create(
        platform="telegram",
        user_id=uid_str,
        username=username or "",
        label=label or "",
        first_message=first_message or "",
    )
    if req is None:
        # Hit the per-platform cap — refuse hard.
        return AccessDecision(
            allowed=False, role="unknown",
            reason="pairing queue full (too many pending requests)",
        )
    return AccessDecision(
        allowed=False, role="unknown",
        reason="new pairing request created — owner notified",
        pairing_code=req.code,
        pairing_pending=False,
    )


# ─── Grant / revoke ─────────────────────────────────────────────────


def grant_telegram_access(
    user_id: int | str,
    role: str = "trusted",
    label: str = "",
    *,
    ttl_seconds: Optional[float] = None,
) -> dict:
    """Atomically grant a Telegram user access. Updates BOTH:
      - `roles.json::speakers.<id>.role = role`
      - `channels.json::allowed_users` (numeric ID added) for
        backwards-compat with any installs that still gate via
        the legacy list.

    `ttl_seconds` — optional auto-revoke timer used by the
    inline-keyboard pairing flow's "Allow Once (1h)" /
    "Allow Session (24h)" buttons. When set, the grant carries an
    `expires_at` field in roles.json that `roles.role_of` honours
    by lazy-revoking back to guest. Channels.json::allowed_users is
    NOT removed by the timer (no scheduled job here); the role
    check inside `is_telegram_allowed` is what actually blocks the
    user after expiry.

    Returns a dict the tool layer can serialise — `{ok, user_id,
    speaker_id, role, label, updated_files, note?, expires_at?}`.
    """
    from . import roles as _roles
    uid = str(user_id)
    speaker_id = f"telegram:{uid}"

    if role not in ("owner", "trusted", "guest"):
        return {
            "ok": False,
            "error": f"role must be owner/trusted/guest, got {role!r}",
        }

    updated = []
    expires_at_epoch: Optional[float] = None
    if ttl_seconds and ttl_seconds > 0:
        expires_at_epoch = time.time() + float(ttl_seconds)

    # 1) roles.json — semantic role
    try:
        _roles.set_role(
            speaker_id, role,
            label=label or None,
            expires_at=expires_at_epoch,
        )
        updated.append("roles.json")
    except Exception as e:
        return {
            "ok": False,
            "error": f"roles.json write failed: {type(e).__name__}: {e}",
        }

    # 2) channels.json::allowed_users — legacy filter. Best-effort:
    # find the Telegram channel(s) and append the user id.
    try:
        from . import channels as _channels
        cfg_path = _channels._resolve_channels_path()
        chs = _channels._load_channels()
        changed = False
        for ch in chs:
            if ch.get("type") != "telegram":
                continue
            config = ch.setdefault("config", {})
            allowed = config.get("allowed_users") or []
            if uid not in allowed and username_token(uid) not in allowed:
                allowed.append(uid)
                config["allowed_users"] = allowed
                changed = True
        if changed:
            _channels._save_channels(chs)
            updated.append("channels.json")
    except Exception as e:
        log.warning("grant_telegram_access: channels.json sync failed: %s", e)

    # 3) Clear any pending pairing for this user — they've been
    # approved, don't keep the request around.
    try:
        PAIRING_STORE.consume(uid, platform="telegram")
    except Exception:
        pass

    # Build a human-readable note so the verifier can match claims
    # like "added X to channels.json" against this exact text instead
    # of trying to infer it from the `updated_files` array alone.
    files_str = " and ".join(updated) if updated else "no files"
    ttl_clause = ""
    if expires_at_epoch:
        ttl_clause = (
            f" — auto-revoke at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expires_at_epoch))}"
        )
    note = (
        f"granted role={role} to telegram:{uid}"
        f"{' (label=' + label + ')' if label else ''}"
        f" — added/promoted in {files_str}"
        f"{ttl_clause}"
    )
    result = {
        "ok": True,
        "user_id": uid,
        "speaker_id": speaker_id,
        "role": role,
        "label": label,
        "updated_files": updated,
        "note": note,
    }
    if expires_at_epoch:
        result["expires_at"] = expires_at_epoch
    return result


def username_token(s: str) -> str:
    """Strip '@' prefix from usernames for canonical comparison."""
    return s.lstrip("@") if s else ""


def revoke_telegram_access(user_id: int | str) -> dict:
    """Symmetric counterpart of grant_telegram_access — removes
    from both files."""
    from . import roles as _roles
    uid = str(user_id)
    speaker_id = f"telegram:{uid}"
    updated = []
    try:
        _roles.set_role(speaker_id, "guest")
        updated.append("roles.json")
    except Exception as e:
        return {"ok": False, "error": f"roles.json write failed: {e}"}
    try:
        from . import channels as _channels
        chs = _channels._load_channels()
        changed = False
        for ch in chs:
            if ch.get("type") != "telegram":
                continue
            config = ch.setdefault("config", {})
            allowed = config.get("allowed_users") or []
            new = [a for a in allowed if a != uid and a != username_token(uid)]
            if new != allowed:
                config["allowed_users"] = new
                changed = True
        if changed:
            _channels._save_channels(chs)
            updated.append("channels.json")
    except Exception as e:
        log.warning("revoke_telegram_access: channels.json sync failed: %s", e)
    files_str = " and ".join(updated) if updated else "no files"
    note = (
        f"revoked telegram:{uid} — role set to guest, "
        f"removed from {files_str}"
    )
    return {
        "ok": True,
        "user_id": uid,
        "speaker_id": speaker_id,
        "role": "guest",
        "updated_files": updated,
        "note": note,
    }


def approve_pairing(
    code_or_user_id: str,
    label: str = "",
    *,
    role: str = "trusted",
    ttl_seconds: Optional[float] = None,
) -> dict:
    """Owner-side approval: look up the pending request by code or
    by user_id, grant access, clear the request.

    Optional knobs (used by the inline-keyboard buttons in
    channels.py):
      - `role`     — 'trusted' (default), 'guest' (for short
                     one-time grants), or 'owner' (only if the
                     owner explicitly wants a co-owner).
      - `ttl_seconds` — auto-revoke timer in seconds. 3600 = "Allow
                     Once" (1h, guest role), 86400 = "Session" (24h,
                     trusted role), None = "Always" (no timer)."""
    req = PAIRING_STORE.consume(code_or_user_id, platform="telegram")
    if req is None:
        return {
            "ok": False,
            "error": f"no pending pairing request for {code_or_user_id!r}",
            "pending": [r.to_dict() for r in PAIRING_STORE.list_pending()],
        }
    use_label = label or req.label or req.username
    res = grant_telegram_access(
        req.user_id, role=role, label=use_label, ttl_seconds=ttl_seconds,
    )
    res["pairing_request"] = req.to_dict()
    ttl_clause = f" (TTL: {int(ttl_seconds)}s)" if ttl_seconds else ""
    res["note"] = (
        f"granted role={role} access to {use_label or req.user_id} "
        f"via pairing code {req.code}{ttl_clause}"
    )
    return res


# ─── Inline-keyboard callback bridge ────────────────────────────────


def _register_pairing_callback() -> None:
    """Wire the pairing flow into the tg_interactive callback
    dispatcher so channels.py doesn't need to know about access.py's
    internals. Called once at module import (idempotent because the
    dispatcher's register is idempotent).

    callback_data shapes handled here:
      - pair:approve:once:<code>     (guest, 1h TTL)
      - pair:approve:session:<code>  (trusted, 24h TTL)
      - pair:approve:always:<code>   (trusted, no TTL)
      - pair:deny:<code>             (drop request)
    """
    from . import tg_interactive as _tg
    from . import roles as _roles_mod

    def _handler(parts: list[str], ctx: dict) -> _tg.CallbackResult:
        if not parts:
            return _tg.CallbackResult(ok=False, toast="malformed callback")

        # Owner-only: the bot owner is the only speaker who should
        # be able to press these. Anyone else just sees a toast.
        clicker_id = ctx.get("clicker_speaker_id") or ""
        if not _roles_mod.is_owner(clicker_id):
            return _tg.CallbackResult(
                ok=False,
                toast="only the owner can approve pairing requests",
                clear_keyboard=False,
            )

        action = parts[0]
        if action == "approve":
            if len(parts) < 3:
                return _tg.CallbackResult(ok=False, toast="malformed approve")
            mode, code = parts[1], parts[2]
            mode_map = {
                "once":    {"role": "guest",   "ttl_seconds": 3600.0,    "label": "Allow once"},
                "session": {"role": "trusted", "ttl_seconds": 86400.0,   "label": "Session"},
                "always":  {"role": "trusted", "ttl_seconds": None,      "label": "Always"},
            }
            mode_cfg = mode_map.get(mode)
            if mode_cfg is None:
                return _tg.CallbackResult(ok=False, toast=f"unknown mode {mode!r}")
            res = approve_pairing(
                code,
                role=mode_cfg["role"],
                ttl_seconds=mode_cfg["ttl_seconds"],
            )
            if not res.get("ok"):
                return _tg.CallbackResult(
                    ok=False,
                    toast=res.get("error") or "approve failed",
                    clear_keyboard=False,
                )
            who = res.get("label") or res.get("user_id")
            ttl_clause = (
                f" (auto-revoke in {int(mode_cfg['ttl_seconds'])//3600}h)"
                if mode_cfg["ttl_seconds"] else ""
            )
            return _tg.CallbackResult(
                ok=True,
                edited_text=(
                    f"✅ <b>Approved</b> — {_tg.escape_html(str(who))} "
                    f"granted role=<b>{mode_cfg['role']}</b>"
                    f"{ttl_clause}.\n<i>Mode: {mode_cfg['label']}</i>"
                ),
                toast=f"approved ({mode_cfg['label']})",
            )
        if action == "deny":
            if len(parts) < 2:
                return _tg.CallbackResult(ok=False, toast="malformed deny")
            code = parts[1]
            res = deny_pairing(code)
            if not res.get("ok"):
                return _tg.CallbackResult(
                    ok=False,
                    toast=res.get("error") or "deny failed",
                    clear_keyboard=False,
                )
            who = res.get("label") or res.get("user_id")
            return _tg.CallbackResult(
                ok=True,
                edited_text=(
                    f"❌ <b>Denied</b> — pairing request from "
                    f"{_tg.escape_html(str(who))} dropped."
                ),
                toast="denied",
            )
        return _tg.CallbackResult(ok=False, toast=f"unknown action {action!r}")

    _tg.register_callback_handler("pair", _handler)


# Auto-wire on module import (idempotent — the dispatcher overwrites).
_register_pairing_callback()


def deny_pairing(code_or_user_id: str) -> dict:
    """Owner-side rejection: drop the pending request without
    granting anything. Symmetric counterpart of approve_pairing.

    Currently no blocklist (the same stranger could re-pair from
    the next message); a blocklist could land later as a follow-up.
    """
    req = PAIRING_STORE.consume(code_or_user_id, platform="telegram")
    if req is None:
        return {
            "ok": False,
            "error": f"no pending pairing request for {code_or_user_id!r}",
        }
    return {
        "ok": True,
        "user_id": req.user_id,
        "speaker_id": req.requester_speaker_id,
        "label": req.label,
        "code": req.code,
        "note": f"denied pairing code {req.code} for {req.label or req.user_id}",
    }


def list_pending_pairings() -> list[dict]:
    return [r.to_dict() for r in PAIRING_STORE.list_pending()]


# ─── One-shot legacy migration ──────────────────────────────────────


_MIGRATION_MARKER_KEY = "_access_migration_done"


def migrate_legacy_allowed_users() -> dict:
    """One-shot: copy every numeric Telegram id from
    `channels.json::allowed_users` into `roles.json::trusted`.
    Idempotent — marker stored in roles.json so subsequent boots
    skip the work."""
    from . import roles as _roles
    try:
        state = _roles._load()
    except Exception as e:
        return {"ok": False, "error": f"roles load failed: {e}"}
    if state.get(_MIGRATION_MARKER_KEY):
        return {"ok": True, "migrated": 0, "note": "already done"}

    migrated = 0
    try:
        from . import channels as _channels
        chs = _channels._load_channels()
    except Exception as e:
        return {"ok": False, "error": f"channels load failed: {e}"}

    for ch in chs:
        if ch.get("type") != "telegram":
            continue
        for entry in (ch.get("config", {}).get("allowed_users") or []):
            s = str(entry).strip().lstrip("@")
            if not s.isdigit():
                continue  # skip usernames; only IDs are speaker-safe
            speaker_id = f"telegram:{s}"
            existing_role = _roles.role_of(speaker_id)
            if existing_role in ("owner", "trusted"):
                continue
            try:
                _roles.set_role(speaker_id, "trusted")
                migrated += 1
            except Exception as e:
                log.warning("migrate: set_role(%s) failed: %s", speaker_id, e)

    # Mark done
    state = _roles._load()
    state[_MIGRATION_MARKER_KEY] = True
    try:
        _roles._save(state)
    except Exception:
        pass

    return {"ok": True, "migrated": migrated}
