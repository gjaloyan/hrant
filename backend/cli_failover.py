"""`hrant failover` subcommand group.

Extracted from cli.py per audit #21. Handles the multi-provider
failover chain (Phase 15B): status / enable / disable / add /
remove / clear.
"""
from __future__ import annotations

import argparse


def _print_ok(msg: str) -> None:
    from .cli import _print_ok as f
    f(msg)


def _print_err(msg: str) -> None:
    from .cli import _print_err as f
    f(msg)


def cmd_failover_status(args: argparse.Namespace) -> int:
    """`hrant failover status` — show current chain + counts of
    recent attempts. Inspectable from the CLI without spinning up
    the WebUI."""
    from . import failover as _fo
    from .cli_colors import c
    cfg = _fo.load_config()
    print()
    print(c.heading("  Failover chain"))
    print(f"  {c.muted('enabled:')}      {c.success('yes') if cfg['enabled'] else c.muted('no')}")
    print(f"  {c.muted('max_attempts:')} {cfg.get('max_attempts', 4)}")
    print(f"  {c.muted('retry on:')}     {', '.join(cfg.get('retry_on') or [])}")
    print()
    chain = cfg.get("chain") or []
    if not chain:
        print(f"  {c.muted('(chain is empty — add providers via WebUI Providers tab')}")
        print(f"  {c.muted(' or `hrant failover add <provider_id> <model>`)')}")
        print()
        return 0
    print(f"  {c.muted('order:')}")
    for i, entry in enumerate(chain, start=1):
        pid = entry.get("provider_id", "?")
        model = entry.get("model", "?")
        print(f"    {i}) {c.accent(pid)} / {c.success(model)}")
    print()
    return 0


def cmd_failover_enable(args: argparse.Namespace) -> int:
    from . import failover as _fo
    cfg = _fo.load_config()
    cfg["enabled"] = True
    _fo.save_config(cfg)
    _print_ok("failover enabled")
    return 0


def cmd_failover_disable(args: argparse.Namespace) -> int:
    from . import failover as _fo
    cfg = _fo.load_config()
    cfg["enabled"] = False
    _fo.save_config(cfg)
    _print_ok("failover disabled")
    return 0


def cmd_failover_add(args: argparse.Namespace) -> int:
    """Append a (provider, model) pair to the end of the chain.

    Validates the provider exists AND (best-effort) that the model
    is one of the provider's declared `models` — otherwise the
    failover will silently skip this entry at runtime because
    create_llm couldn't find it. Pass `--force` to bypass the model
    check for providers whose model list isn't pre-discovered
    (e.g. fresh Ollama install)."""
    from . import failover as _fo
    from .providers import get_provider
    from .cli_colors import c
    provider = get_provider(args.provider_id)
    if not provider:
        _print_err(f"no provider with id '{args.provider_id}' "
                   "(see `hrant provider list`)")
        return 1
    declared_models = list(provider.get("models") or [])
    default_model = provider.get("default_model") or ""
    valid_models = set(declared_models)
    if default_model:
        valid_models.add(default_model)
    if valid_models and args.model not in valid_models and not args.force:
        _print_err(
            f"model '{args.model}' is not in the provider's declared "
            f"list ({sorted(valid_models)[:5]}{'...' if len(valid_models) > 5 else ''}). "
            "Use --force to add anyway, or check the model name."
        )
        return 1
    cfg = _fo.load_config()
    chain = list(cfg.get("chain") or [])
    chain.append({"provider_id": args.provider_id, "model": args.model})
    cfg["chain"] = chain
    saved = _fo.save_config(cfg)
    _print_ok(
        f"chain now has {len(saved['chain'])} entries: "
        + " " + c.muted("→") + " ".join(f"{e['provider_id']}/{e['model']}" for e in saved['chain'])
    )
    return 0


def cmd_failover_remove(args: argparse.Namespace) -> int:
    """Remove chain entry at the given 1-based index."""
    from . import failover as _fo
    cfg = _fo.load_config()
    chain = list(cfg.get("chain") or [])
    idx = args.index - 1
    if not (0 <= idx < len(chain)):
        _print_err(f"index {args.index} out of range (chain has {len(chain)})")
        return 1
    removed = chain.pop(idx)
    cfg["chain"] = chain
    _fo.save_config(cfg)
    _print_ok(f"removed {removed['provider_id']}/{removed['model']}")
    return 0


def cmd_failover_clear(args: argparse.Namespace) -> int:
    from . import failover as _fo
    cfg = _fo.load_config()
    cfg["chain"] = []
    _fo.save_config(cfg)
    _print_ok("chain cleared")
    return 0
