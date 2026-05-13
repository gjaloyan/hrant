import { useEffect, useState } from "react";
import IdentityEditor from "./IdentityEditor";
import {
  IdentityProfileMeta,
  fetchIdentity,
  fetchIdentityProfiles,
  updateIdentity,
} from "../../api";

type HistoryVersion = { timestamp: string; path: string; size: number };

type Props = {
  userProfile: string;
  setUserProfile: (v: string) => void;
  dirty: boolean;
  onSave: (file: string, content: string) => void;
  onReload: () => void;
  setDirty: (d: boolean) => void;
  history: HistoryVersion[];
};

const DEFAULT_SPEAKER = "webui:default";

export default function UserProfileTab({
  userProfile, setUserProfile, dirty, onSave, onReload, setDirty, history,
}: Props) {
  const [profiles, setProfiles] = useState<IdentityProfileMeta[]>([]);
  const [speakerId, setSpeakerId] = useState<string>(DEFAULT_SPEAKER);
  // Local copy of the non-default-speaker profile body. The default
  // speaker uses the parent's `userProfile` state for back-compat
  // with the existing dirty/save flow.
  const [perSpeakerText, setPerSpeakerText] = useState<string>("");
  const [perSpeakerDirty, setPerSpeakerDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const flash = (text: string) => {
    setMsg(text);
    setTimeout(() => setMsg(""), 3500);
  };

  const reloadProfiles = async () => {
    try {
      const r = await fetchIdentityProfiles();
      setProfiles(r.profiles);
    } catch (e: any) {
      flash("Load profiles failed: " + e.message);
    }
  };

  useEffect(() => {
    reloadProfiles();
  }, []);

  const switchSpeaker = async (sid: string) => {
    if (sid === speakerId) return;
    if ((sid === DEFAULT_SPEAKER ? dirty : perSpeakerDirty) &&
        !confirm("You have unsaved changes. Switch anyway?")) {
      return;
    }
    setSpeakerId(sid);
    if (sid === DEFAULT_SPEAKER) {
      // Parent already has the default speaker's text loaded.
      setPerSpeakerText("");
      setPerSpeakerDirty(false);
      return;
    }
    setBusy(true);
    try {
      const r = await fetchIdentity(sid);
      setPerSpeakerText(r.user_profile || "");
      setPerSpeakerDirty(false);
    } catch (e: any) {
      flash("Load failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleSavePerSpeaker = async () => {
    setBusy(true);
    try {
      await updateIdentity("user", perSpeakerText, speakerId);
      flash(`Saved ${speakerId}.`);
      setPerSpeakerDirty(false);
      reloadProfiles();
    } catch (e: any) {
      flash("Save failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const isDefault = speakerId === DEFAULT_SPEAKER;

  return (
    <div className="flex flex-col h-full">
      {/* Speaker selector */}
      <div className="flex items-center gap-2 mb-3 bg-slate-800/60 rounded p-2 text-xs">
        <span className="opacity-60">Speaker:</span>
        <select
          value={speakerId}
          onChange={(e) => switchSpeaker(e.target.value)}
          disabled={busy}
          className="flex-1 bg-slate-900 rounded px-2 py-1 outline-none focus:ring-1 focus:ring-sky-600"
        >
          <option value={DEFAULT_SPEAKER}>{DEFAULT_SPEAKER} (WebUI user)</option>
          {profiles
            .filter((p) => p.speaker_id !== DEFAULT_SPEAKER)
            .map((p) => (
              <option key={p.speaker_id} value={p.speaker_id}>
                {p.speaker_id} · {p.modified.slice(0, 16)}
              </option>
            ))}
        </select>
        {msg && <span className="text-amber-300 text-[11px]">{msg}</span>}
      </div>

      <div className="text-[11px] text-slate-500 mb-2">
        Per-speaker user_profile lives in{" "}
        <span className="font-mono">
          knowledge/identity/
          {isDefault ? "user.md" : `profiles/${speakerId.replace(":", "_")}.md`}
        </span>.
        Each speaker (each Telegram user, WebUI user, …) has independent
        memory + profile so context never bleeds between conversations.
      </div>

      {isDefault ? (
        <IdentityEditor
          file="user"
          value={userProfile}
          setter={setUserProfile}
          dirty={dirty}
          onSave={onSave}
          onReload={onReload}
          setDirty={setDirty}
        />
      ) : (
        <div className="flex flex-col flex-1 min-h-0">
          <textarea
            value={perSpeakerText}
            onChange={(e) => {
              setPerSpeakerText(e.target.value);
              setPerSpeakerDirty(true);
            }}
            className="flex-1 bg-slate-900 rounded p-3 text-xs font-mono outline-none resize-none focus:ring-1 focus:ring-sky-600"
            disabled={busy}
          />
          <div className="flex gap-2 mt-2 text-xs">
            <button
              onClick={handleSavePerSpeaker}
              disabled={busy || !perSpeakerDirty}
              className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1.5"
            >
              Save {speakerId}
            </button>
            <button
              onClick={() => switchSpeaker(speakerId)}
              disabled={busy}
              className="bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded px-3 py-1.5"
            >
              Reload
            </button>
            {perSpeakerDirty && (
              <span className="text-amber-300 self-center">● unsaved</span>
            )}
          </div>
        </div>
      )}

      {history.length > 0 && isDefault && (
        <details className="mt-3">
          <summary className="text-xs cursor-pointer opacity-60">
            Profile history ({history.length} versions)
          </summary>
          <div className="mt-1 space-y-0.5 text-xs max-h-32 overflow-y-auto">
            {history.map((v, i) => (
              <div key={i} className="flex gap-2 opacity-60">
                <span>{v.timestamp}</span>
                <span>{v.size}b</span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
