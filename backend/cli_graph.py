"""`hrant graph` subcommand group (Phase 16C knowledge graph).

Extracted from cli.py per audit #21. Handles the kgraph
surface: stats / search / show / rebuild.
"""
from __future__ import annotations

import argparse


def _print_ok(msg: str) -> None:
    from .cli import _print_ok as f
    f(msg)


def _print_err(msg: str) -> None:
    from .cli import _print_err as f
    f(msg)


def cmd_graph_stats(args: argparse.Namespace) -> int:
    """`hrant graph stats` — totals + top topics."""
    from .graph import query as _gq
    from .cli_colors import c
    s = _gq.stats()
    print()
    print(c.heading("  Knowledge graph"))
    print()
    print(f"  {c.muted('total nodes:'):<18} {s['total_nodes']}")
    print(f"  {c.muted('total edges:'):<18} {s['total_edges']}")
    print()
    print(c.muted("  by kind:"))
    for kind, count in s["by_kind"].items():
        if count > 0:
            print(f"    {kind:<10}  {count}")
    if s["top_topics"]:
        print()
        print(c.muted("  top topics:"))
        for t in s["top_topics"]:
            print(f"    {c.accent(t['label']):<32}  {c.muted(str(t['degree']) + ' connections')}")
    print()
    return 0


def cmd_graph_search(args: argparse.Namespace) -> int:
    from .graph import query as _gq
    from .cli_colors import c
    results = _gq.search(args.query, kind=args.kind, limit=args.limit)
    if not results:
        print(c.muted("  no matches"))
        return 0
    print()
    print(c.heading(f"  Search: '{args.query}'  ({len(results)} results)"))
    print()
    for r in results:
        kind_c = {
            "fact": c.success(r["kind"]),
            "topic": c.accent(r["kind"]),
            "skill": c.warn(r["kind"]),
            "project": c.info(r["kind"]),
            "entity": c.muted(r["kind"]),
        }.get(r["kind"], r["kind"])
        deg = c.muted(f"  ({r.get('degree', 0)} conn)")
        print(f"  [{kind_c}] {r['label'][:100]}{deg}")
    print()
    return 0


def cmd_graph_show(args: argparse.Namespace) -> int:
    from .graph import query as _gq
    from .cli_colors import c
    n = _gq.neighborhood(args.node_id)
    if n is None:
        _print_err(f"no node with id '{args.node_id}'")
        return 1
    node = n["node"]
    print()
    print(c.heading(f"  {node['label']}"))
    print(f"  {c.muted('id:'):<14} {node['id']}")
    print(f"  {c.muted('kind:'):<14} {node['kind']}")
    print(f"  {c.muted('weight:'):<14} {node['weight']}")
    if node.get("metadata"):
        print(f"  {c.muted('metadata:'):<14} {node['metadata']}")
    print()
    print(c.muted(f"  neighbors ({n['neighbor_count']}):"))
    for entry in n["neighbors"]:
        e = entry["edge"]
        o = entry["node"]
        arrow = "→" if entry["direction"] == "out" else "←"
        kind_c = {
            "fact": c.success,
            "topic": c.accent,
            "skill": c.warn,
            "project": c.info,
            "entity": c.muted,
        }.get(o["kind"], lambda x: x)
        print(f"    {arrow} {c.muted(e['kind']):<14} [{kind_c(o['kind'])}] {o['label'][:80]}")
    print()
    return 0


def cmd_graph_rebuild(args: argparse.Namespace) -> int:
    from .graph import builder as _gb
    stats = _gb.rebuild()
    _print_ok(
        f"graph rebuilt: {stats['facts']} facts, {stats['topics']} topics, "
        f"{stats['skills']} skills, {stats['projects']} projects, "
        f"{stats['edges']} edges"
    )
    return 0
