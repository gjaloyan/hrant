---
name: summarize_pdf
description: Read a local PDF/DOCX/MD file and produce a structured summary.
triggers: [pdf, docx, документ, конспект, summarize, summary, резюме, изложи, реферат]
when_to_use: |
  User asks to summarize, recap, extract key points from, or analyze a local
  document file (PDF, DOCX, MD, TXT). Triggered both by the file extension
  and by Russian/English summarization verbs.
---

# How to summarize a document

## Steps
1. Call `read_file(path=...)` with the path the user gave.
   If the user mentioned a file but didn't give a path, ask them for it first.
2. If the document is large, the result is already truncated to 20k chars.
   Don't loop calling read_file with bigger limits unless the user asks.
3. Produce a summary in this exact structure:

   ```
   ## Документ
   <название файла, тип, примерный объём>

   ## Главное в одну фразу
   <один абзац, 1–2 предложения>

   ## Ключевые тезисы
   - <bullet>
   - <bullet>
   - <bullet>
   (5–8 пунктов)

   ## Цифры и факты
   - <если есть>

   ## Выводы / рекомендации
   - <если применимо>
   ```

## Style
- Используй язык документа, не свой по умолчанию.
- Не выдумывай факты, которых нет в извлечённом тексте.
- Если документ обрезан — честно отметь "(текст обрезан до 20к символов)".
