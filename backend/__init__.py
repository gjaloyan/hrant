"""Backend package init.

Side effect: register all built-in tools into the global ToolRegistry.
Skill loader and MCP client also register their tools here once they are
constructed (lazily, on first agent run, to avoid import cycles).
"""
from .builtin_tools import register_builtin_tools

register_builtin_tools()
