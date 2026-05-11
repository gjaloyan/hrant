---
module: backend/api/attachments.py
category: self
kind: module
updated: 2026-05-02T08:54:15.136342+00:00
source_mtime: 2026-04-30T10:15:41.482214+00:00
loc: 162
truncated: false
---

# backend/api/attachments.py

## Purpose
Defines a FastAPI router for attachment storage, retrieval, metadata listing, and speech-to-text transcription. Uploaded attachments are stored through the shared attachment service with sha256-based deduplication, while audio transcription is delegated to the configured transcriber backend and can optionally persist the audio plus transcript as an attachment.

## Public interface
- `router` (constant) - FastAPI APIRouter containing attachment and transcription routes.
- `upload_attachment` (function) - POST endpoint that uploads and stores a non-empty file attachment, returning its metadata.
- `attachment_meta` (function) - GET endpoint that returns metadata for an attachment by sha256.
- `attachment_blob` (function) - GET endpoint that returns raw attachment bytes with the stored MIME type.
- `list_attachments` (function) - GET endpoint that returns all attachment metadata records and storage stats.
- `transcribe` (function) - POST endpoint that transcribes uploaded audio and optionally saves it as an audio attachment.
- `transcribe_status` (function) - GET endpoint that returns the current transcriber status.
- `TranscriberConfigUpdate` (class) - Pydantic request model for updating transcriber backend configuration.
- `transcribe_config_put` (function) - PUT endpoint that saves transcriber configuration, resets the transcriber, and returns updated status.
- `transcribe_reset` (function) - POST endpoint that resets the transcriber and returns its status.

## Dependencies
- backend.attachments
- backend.transcriber

## Notes
Empty uploads are rejected with HTTP 400, and missing attachment metadata or blobs return HTTP 404. Transcription failure returns HTTP 503 with backend status details. Attachment response dictionaries are manually duplicated across several endpoints, so metadata field changes must be kept in sync.
