---
module: backend/autonomic/events.py
category: self
kind: module
updated: 2026-04-27T11:17:45.337536+00:00
source_mtime: 2026-04-16T20:09:53.454955+00:00
loc: 44
truncated: false
---

# backend/autonomic/events.py

## Purpose
This module implements a simple in-process event bus for autonomic coordination, allowing synchronous event publishing where all subscribers are invoked immediately during the publish call. It ensures that exceptions in one subscriber do not prevent others from running.

## Public interface
- `EventBus` (class) - A class that manages event subscriptions and publishing.
- `subscribe` (function) - Subscribes a handler to a specific topic and returns a token.
- `unsubscribe` (function) - Unsubscribes a handler using its token.
- `publish` (function) - Publishes an event to all handlers subscribed to a topic.

## Dependencies
(none)

## Notes
The module is straightforward, focusing on synchronous event handling with a simple subscription mechanism. It handles exceptions in subscribers gracefully by logging them, ensuring that other subscribers are not affected. The use of dataclasses simplifies the implementation of internal data structures.
