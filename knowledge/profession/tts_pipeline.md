---
topic: TTS pipeline
category: profession
created: 2026-05-07 16:52
updated: 2026-05-07 16:52
keywords: TTS pipeline, streaming synthesis, text-to-speech, voice agent, STT LLM TTS, latency, PCM, MP3, voice cloning, Realtime API
source: https://livekit.com/blog/voice-agent-architecture-stt-llm-tts-pipelines-explained; https://inworld.ai/resources/build-stt-llm-tts-voice-pipeline; https://huggingface.co/docs/transformers/tasks/text-to-speech
confidence: partial
access_count: 6
---

# TTS pipeline

## Что это
TTS pipeline — часть голосового агента, которая преобразует текстовый ответ LLM в аудио и воспроизводит его пользователю. В классической схеме работает как STT → LLM → TTS.

## Ключевые параметры
- Основные критерии TTS: качество голоса, latency, скорость начала генерации аудио.
- Критичный режим для voice agents: streaming synthesis — генерация аудио по частичному тексту.
- Sequential pipeline даёт задержку ответа примерно 2–4 секунды.
- Streaming pipeline при потоковой STT + LLM + TTS может дать end-to-end latency < 1 секунды.
- Пример audio config:
  - MP3 или PCM
  - sampleRateHertz: 24000
- Hugging Face TTS pipeline: `text-to-audio` / alias `text-to-speech`.
- Примеры моделей TTS в Transformers: Dia, CSM, Bark, MMS, VITS, SpeechT5.
- Некоторые TTS модели поддерживают voice cloning через reference audio.
- Некоторые модели могут генерировать non-verbal audio: смех, вздохи, плач, музыку.

## Практические заметки
- Для real-time диалогов использовать streaming TTS, а не ожидать полного ответа LLM.
- LLM должен стримить токены, чтобы TTS начал синтез до завершения генерации текста.
- Практичный буфер: собирать токены LLM до границы предложения и сразу отправлять первое готовое предложение в TTS.
- Для batch-сценариев подходит sync pipeline: записать аудио → STT → LLM → TTS → сохранить файл.
- Для production voice agents предпочтительна потоковая архитектура или единый Realtime API/WebSocket.
- PCM удобен для потокового воспроизведения; MP3 — для сохранения готового аудиофайла.
- При выборе TTS учитывать не только качество голоса, но и time-to-first-audio.

## Частые ошибки
- Ожидание полного ответа LLM перед TTS → высокая задержка; исправление: включить streaming LLM + streaming TTS.
- Sequential pipeline для живого диалога → неестественная пауза 2–4 секунды; исправление: перекрывать STT, LLM и TTS.
- Отправка слишком мелких фрагментов текста в TTS → рваная речь; исправление: буферизовать до границ предложений.
- Использование batch TTS endpoint для real-time → поздний старт аудио; исправление: использовать streaming endpoint.
- Неподходящий audio encoding/sample rate → проблемы воспроизведения; исправление: явно задать PCM/MP3 и sample rate, например 24000 Hz.

## Причинно-следственные связи
- Sequential STT → LLM → TTS causes accumulated latency
- Full-response wait before TTS causes unnatural conversation delay
- Streaming LLM tokens enables early TTS synthesis
- Streaming TTS enables lower perceived response time
- Sentence-boundary buffering enables smoother audio playback
- One WebSocket Realtime API enables simpler production voice pipeline
- PCM streaming enables chunk-by-chunk playback
- Reference audio enables voice cloning

## Связанные темы
- [[STT pipeline]]
- [[LLM streaming]]
- [[Voice agent architecture]]
- [[Realtime API]]
- [[Turn detection]]
- [[Voice cloning]]
