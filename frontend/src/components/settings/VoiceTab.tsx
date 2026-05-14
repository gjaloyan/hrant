import { useEffect, useState } from "react";
import {
  DiscoverResult,
  TranscriberConfig,
  TranscriberStatus,
  TtsConfig,
  TtsStatus,
  fetchTranscribeConfig,
  fetchTranscribeStatus,
  fetchTtsConfig,
  fetchTtsStatus,
  putTranscribeConfig,
  putTtsConfig,
  resetTranscriber,
  resetTts,
  runDiscover,
} from "../../api";

type Props = { flash: (msg: string) => void };

const STT_BACKENDS: { id: NonNullable<TranscriberConfig["backend"]>; label: string }[] = [
  { id: "auto", label: "Auto (local_whisper → whisper_cpp → openai)" },
  { id: "local_whisper", label: "Local FastAPI Whisper wrapper" },
  { id: "whisper_cpp", label: "whisper.cpp REST server" },
  { id: "openai_whisper", label: "OpenAI Whisper API" },
  { id: "disabled", label: "Disabled (no STT, text-only inputs)" },
];

const TTS_BACKENDS: { id: NonNullable<TtsConfig["backend"]>; label: string }[] = [
  { id: "auto", label: "Auto (edge_tts → local_piper → openai) — recommended" },
  { id: "edge_tts", label: "Edge TTS (Microsoft, free, online, ~400 voices)" },
  { id: "local_piper", label: "Local Piper HTTP server" },
  { id: "openai_tts", label: "OpenAI TTS" },
  { id: "disabled", label: "Disabled (text-only replies)" },
];

export default function VoiceTab({ flash }: Props) {
  // Loaded state
  const [sttStatus, setSttStatus] = useState<TranscriberStatus | null>(null);
  const [ttsStatus, setTtsStatus] = useState<TtsStatus | null>(null);
  const [busy, setBusy] = useState(false);

  // STT form state
  const [sttBackend, setSttBackend] = useState<NonNullable<TranscriberConfig["backend"]>>("auto");
  const [localWhisperUrl, setLocalWhisperUrl] = useState("");
  const [localWhisperModel, setLocalWhisperModel] = useState("whisper-medium");
  const [whisperCppUrl, setWhisperCppUrl] = useState("");
  const [openaiWhisperModel, setOpenaiWhisperModel] = useState("whisper-1");

  // TTS form state
  const [ttsBackend, setTtsBackend] = useState<NonNullable<TtsConfig["backend"]>>("auto");
  const [piperUrl, setPiperUrl] = useState("");
  const [piperVoice, setPiperVoice] = useState("en_US-lessac-medium");
  const [piperVoiceRu, setPiperVoiceRu] = useState("ru_RU-irina-medium");
  const [edgeVoice, setEdgeVoice] = useState("en-US-AriaNeural");
  const [edgeVoiceRu, setEdgeVoiceRu] = useState("ru-RU-SvetlanaNeural");
  const [openaiTtsModel, setOpenaiTtsModel] = useState("tts-1");
  const [openaiTtsVoice, setOpenaiTtsVoice] = useState("alloy");

  // Discover form
  const [discoverHost, setDiscoverHost] = useState("");
  const [discoverResult, setDiscoverResult] = useState<DiscoverResult | null>(null);

  const refresh = async () => {
    try {
      const [stt, tts, sttCfg, ttsCfg] = await Promise.all([
        fetchTranscribeStatus(),
        fetchTtsStatus(),
        fetchTranscribeConfig().catch(() => ({}) as TranscriberConfig),
        fetchTtsConfig().catch(() => ({}) as TtsConfig),
      ]);
      setSttStatus(stt);
      setTtsStatus(tts);
      // Populate STT form
      if (sttCfg.backend) setSttBackend(sttCfg.backend);
      if (sttCfg.local_whisper?.url) setLocalWhisperUrl(sttCfg.local_whisper.url);
      if (sttCfg.local_whisper?.model) setLocalWhisperModel(sttCfg.local_whisper.model);
      if (sttCfg.whisper_cpp?.url) setWhisperCppUrl(sttCfg.whisper_cpp.url);
      if (sttCfg.openai_whisper?.model) setOpenaiWhisperModel(sttCfg.openai_whisper.model);
      // Populate TTS form
      if (ttsCfg.backend) setTtsBackend(ttsCfg.backend);
      if (ttsCfg.local_piper?.url) setPiperUrl(ttsCfg.local_piper.url);
      if (ttsCfg.local_piper?.voice) setPiperVoice(ttsCfg.local_piper.voice);
      if (ttsCfg.local_piper?.voice_ru) setPiperVoiceRu(ttsCfg.local_piper.voice_ru);
      if (ttsCfg.edge_tts?.voice) setEdgeVoice(ttsCfg.edge_tts.voice);
      if (ttsCfg.edge_tts?.voice_ru) setEdgeVoiceRu(ttsCfg.edge_tts.voice_ru);
      if (ttsCfg.openai_tts?.model) setOpenaiTtsModel(ttsCfg.openai_tts.model);
      if (ttsCfg.openai_tts?.voice) setOpenaiTtsVoice(ttsCfg.openai_tts.voice);
    } catch (e: any) {
      flash("Error loading voice config: " + e.message);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleSaveStt = async () => {
    setBusy(true);
    try {
      const cfg: Partial<TranscriberConfig> = { backend: sttBackend };
      if (sttBackend === "local_whisper" || sttBackend === "auto") {
        cfg.local_whisper = { url: localWhisperUrl.trim(), model: localWhisperModel.trim() };
      }
      if (sttBackend === "whisper_cpp" || sttBackend === "auto") {
        cfg.whisper_cpp = { url: whisperCppUrl.trim() };
      }
      if (sttBackend === "openai_whisper" || sttBackend === "auto") {
        cfg.openai_whisper = { model: openaiWhisperModel.trim() };
      }
      const result = await putTranscribeConfig(cfg);
      setSttStatus(result.transcriber);
      const live = result.transcriber;
      if (live.backend === "disabled") {
        flash(`STT: no backend reachable: ${live.last_error || "see config"}`);
      } else {
        flash(`STT online: ${live.backend} / ${live.model || "?"}`);
      }
    } catch (e: any) {
      flash("STT save failed: " + (e.message || String(e)));
    } finally {
      setBusy(false);
    }
  };

  const handleSaveTts = async () => {
    setBusy(true);
    try {
      const cfg: Partial<TtsConfig> = { backend: ttsBackend };
      if (ttsBackend === "edge_tts" || ttsBackend === "auto") {
        cfg.edge_tts = {
          voice: edgeVoice.trim(),
          voice_ru: edgeVoiceRu.trim(),
        };
      }
      if (ttsBackend === "local_piper" || ttsBackend === "auto") {
        cfg.local_piper = {
          url: piperUrl.trim(),
          voice: piperVoice.trim(),
          voice_ru: piperVoiceRu.trim(),
        };
      }
      if (ttsBackend === "openai_tts" || ttsBackend === "auto") {
        cfg.openai_tts = { model: openaiTtsModel.trim(), voice: openaiTtsVoice.trim() };
      }
      const result = await putTtsConfig(cfg);
      setTtsStatus(result.tts);
      const live = result.tts;
      if (live.backend === "disabled") {
        flash(`TTS: no backend reachable: ${live.last_error || "see config"}`);
      } else {
        flash(`TTS online: ${live.backend} / ${live.voice || "?"}`);
      }
    } catch (e: any) {
      flash("TTS save failed: " + (e.message || String(e)));
    } finally {
      setBusy(false);
    }
  };

  const handleResetStt = async () => {
    setBusy(true);
    try {
      const r = await resetTranscriber();
      setSttStatus(r.transcriber);
      flash(`STT re-probed: ${r.transcriber.backend}`);
    } catch (e: any) {
      flash("Reset failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleResetTts = async () => {
    setBusy(true);
    try {
      const r = await resetTts();
      setTtsStatus(r.tts);
      flash(`TTS re-probed: ${r.tts.backend}`);
    } catch (e: any) {
      flash("Reset failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleDiscover = async (apply: boolean) => {
    setBusy(true);
    try {
      const result = await runDiscover(discoverHost.trim() || undefined, apply);
      setDiscoverResult(result);
      if (result.error) {
        flash(`Discover: ${result.error}`);
      } else {
        const ok = Object.values(result.found || {}).filter((v) => v.ok).length;
        const total = Object.keys(result.found || {}).length;
        flash(`Discover: ${ok}/${total} services up${apply ? " — applied" : ""}`);
        if (apply) {
          // Refresh both STT and TTS forms from the new saved config.
          await refresh();
        }
      }
    } catch (e: any) {
      flash("Discover failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const Dot = ({ status }: { status: string | null | undefined }) => {
    const cls =
      status === "disabled" || !status
        ? "bg-amber-400"
        : "bg-emerald-400";
    return <span className={`inline-block w-2 h-2 rounded-full ${cls}`} />;
  };

  return (
    <div className="space-y-4">
      <h3 className="font-bold">Voice — Speech-to-Text & Text-to-Speech</h3>

      {/* Tailscale / LAN discovery */}
      <div className="bg-slate-800 rounded p-3 space-y-2 text-sm">
        <div className="flex items-center justify-between">
          <span className="font-semibold">Auto-discover services</span>
          <span className="text-xs text-slate-400">
            Probes Whisper/Piper/Ollama on the given host
          </span>
        </div>
        <div className="flex gap-2 items-center">
          <input
            value={discoverHost}
            onChange={(e) => setDiscoverHost(e.target.value)}
            placeholder="host (default: $TAILSCALE_HOST)"
            className="flex-1 bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
          />
          <button
            onClick={() => handleDiscover(false)}
            disabled={busy}
            className="bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Probe
          </button>
          <button
            onClick={() => handleDiscover(true)}
            disabled={busy}
            className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Probe & apply
          </button>
        </div>
        {discoverResult?.error && (
          <div className="text-xs text-rose-300">{discoverResult.error}</div>
        )}
        {discoverResult?.found && (
          <div className="text-xs space-y-0.5 font-mono">
            {Object.entries(discoverResult.found).map(([name, r]) => (
              <div key={name} className="flex gap-2">
                <span className={r.ok ? "text-emerald-300" : "text-rose-300"}>
                  {r.ok ? "✓" : "✗"}
                </span>
                <span className="w-16">{name}</span>
                <span className="text-slate-400">{r.ok ? r.url : r.reason}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* STT block */}
      <div className="bg-slate-800 rounded p-3 space-y-3 text-sm">
        <div className="flex items-center gap-2">
          <Dot status={sttStatus?.backend} />
          <span className="font-semibold">
            STT (Speech-to-Text): <span className="font-mono">{sttStatus?.backend || "—"}</span>
          </span>
          <span className="text-slate-400 text-xs ml-auto">
            model: <span className="font-mono">{sttStatus?.model || "—"}</span>
          </span>
        </div>
        {sttStatus?.last_error && (
          <div className="text-xs text-rose-300 font-mono break-all">{sttStatus.last_error}</div>
        )}
        <div>
          <label className="text-xs text-slate-400 block mb-1">Backend</label>
          <select
            value={sttBackend}
            onChange={(e) => setSttBackend(e.target.value as any)}
            className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600"
          >
            {STT_BACKENDS.map((b) => (
              <option key={b.id} value={b.id}>{b.label}</option>
            ))}
          </select>
        </div>
        {(sttBackend === "local_whisper" || sttBackend === "auto") && (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-slate-400 block mb-1">local_whisper URL</label>
              <input
                value={localWhisperUrl}
                onChange={(e) => setLocalWhisperUrl(e.target.value)}
                placeholder="http://100.x.x.x:8016"
                className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">model</label>
              <input
                value={localWhisperModel}
                onChange={(e) => setLocalWhisperModel(e.target.value)}
                className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
              />
            </div>
          </div>
        )}
        {(sttBackend === "whisper_cpp" || sttBackend === "auto") && (
          <div>
            <label className="text-xs text-slate-400 block mb-1">whisper.cpp URL</label>
            <input
              value={whisperCppUrl}
              onChange={(e) => setWhisperCppUrl(e.target.value)}
              placeholder="http://host:port"
              className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
            />
          </div>
        )}
        {(sttBackend === "openai_whisper" || sttBackend === "auto") && (
          <div>
            <label className="text-xs text-slate-400 block mb-1">OpenAI model</label>
            <input
              value={openaiWhisperModel}
              onChange={(e) => setOpenaiWhisperModel(e.target.value)}
              className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
            />
          </div>
        )}
        <div className="flex gap-2">
          <button
            onClick={handleSaveStt}
            disabled={busy}
            className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Save STT
          </button>
          <button
            onClick={handleResetStt}
            disabled={busy}
            className="bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Re-probe
          </button>
        </div>
      </div>

      {/* TTS block */}
      <div className="bg-slate-800 rounded p-3 space-y-3 text-sm">
        <div className="flex items-center gap-2">
          <Dot status={ttsStatus?.backend} />
          <span className="font-semibold">
            TTS (Text-to-Speech): <span className="font-mono">{ttsStatus?.backend || "—"}</span>
          </span>
          <span className="text-slate-400 text-xs ml-auto">
            voice: <span className="font-mono">{ttsStatus?.voice || "—"}</span>
          </span>
        </div>
        {ttsStatus?.last_error && (
          <div className="text-xs text-rose-300 font-mono break-all">{ttsStatus.last_error}</div>
        )}
        <div>
          <label className="text-xs text-slate-400 block mb-1">Backend</label>
          <select
            value={ttsBackend}
            onChange={(e) => setTtsBackend(e.target.value as any)}
            className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600"
          >
            {TTS_BACKENDS.map((b) => (
              <option key={b.id} value={b.id}>{b.label}</option>
            ))}
          </select>
        </div>
        {(ttsBackend === "edge_tts" || ttsBackend === "auto") && (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-slate-400 block mb-1">
                Edge voice (default)
              </label>
              <input
                value={edgeVoice}
                onChange={(e) => setEdgeVoice(e.target.value)}
                placeholder="en-US-AriaNeural"
                className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">
                Edge voice (Russian)
              </label>
              <input
                value={edgeVoiceRu}
                onChange={(e) => setEdgeVoiceRu(e.target.value)}
                placeholder="ru-RU-SvetlanaNeural"
                className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
              />
            </div>
            <div className="col-span-2 text-[11px] text-slate-500">
              Free Microsoft online TTS. ~400 multilingual voices — full list:{" "}
              <span className="font-mono">python -m edge_tts --list-voices</span>.
              Cyrillic text auto-routes to the Russian voice.
            </div>
          </div>
        )}
        {(ttsBackend === "local_piper" || ttsBackend === "auto") && (
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Piper URL</label>
              <input
                value={piperUrl}
                onChange={(e) => setPiperUrl(e.target.value)}
                placeholder="http://100.x.x.x:8017"
                className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">voice (default)</label>
              <input
                value={piperVoice}
                onChange={(e) => setPiperVoice(e.target.value)}
                className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">voice_ru</label>
              <input
                value={piperVoiceRu}
                onChange={(e) => setPiperVoiceRu(e.target.value)}
                className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
              />
            </div>
          </div>
        )}
        {(ttsBackend === "openai_tts" || ttsBackend === "auto") && (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-slate-400 block mb-1">OpenAI model</label>
              <input
                value={openaiTtsModel}
                onChange={(e) => setOpenaiTtsModel(e.target.value)}
                className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">voice</label>
              <input
                value={openaiTtsVoice}
                onChange={(e) => setOpenaiTtsVoice(e.target.value)}
                className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
              />
            </div>
          </div>
        )}
        <div className="flex gap-2">
          <button
            onClick={handleSaveTts}
            disabled={busy}
            className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Save TTS
          </button>
          <button
            onClick={handleResetTts}
            disabled={busy}
            className="bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Re-probe
          </button>
        </div>
      </div>
    </div>
  );
}
