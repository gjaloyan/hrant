"""Model X — autonomic controller for the agent.

Background subsystem that keeps the agent healthy without conscious intervention
from the cortex (main LLM). Operates through a fixed catalog of levers with
safety classification (green/yellow/red). See
docs/superpowers/specs/2026-04-16-model-x-autonomic-design.md for design.
"""
