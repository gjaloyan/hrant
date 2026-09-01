import { useEffect, useState } from "react";
import {
  SkillFull,
  SkillInstallRequest,
  SkillMeta,
  deleteSkill,
  fetchSkill,
  fetchSkills,
  installSkill,
  reloadSkills,
  setSkillEnabled,
  upsertSkill,
} from "../../api";

type Props = { flash: (msg: string) => void };

const SOURCE_COLOUR: Record<SkillMeta["source"], string> = {
  builtin: "bg-slate-700/60 text-slate-300",
  user: "bg-emerald-700/60 text-emerald-100",
};

const STARTER_MD = `---
name: my_skill
description: One-line description of what this skill does.
triggers: [trigger1, trigger2]
when_to_use: |
  Describe the situation when the agent should use this skill.
---

# My Skill

Detailed instructions for the LLM in free-form Markdown.
Walk through any required steps, the expected response format,
and which tools to call (if any).
`;

export default function SkillsTab({ flash }: Props) {
  const [skills, setSkills] = useState<SkillMeta[]>([]);
  const [userDir, setUserDir] = useState<string>("");
  const [selected, setSelected] = useState<SkillFull | null>(null);
  const [editorText, setEditorText] = useState<string>("");
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);

  // Install dialog state
  const [showInstall, setShowInstall] = useState(false);
  const [installType, setInstallType] = useState<"git" | "zip" | "local">("git");
  const [installSource, setInstallSource] = useState("");
  const [installName, setInstallName] = useState("");
  const [installSubdir, setInstallSubdir] = useState("");

  // New skill dialog
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");

  const refresh = async () => {
    try {
      const r = await fetchSkills();
      setSkills(r.skills);
      setUserDir(r.user_dir);
    } catch (e: any) {
      flash("Load failed: " + e.message);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const openSkill = async (name: string) => {
    if (dirty && !confirm("Discard unsaved changes?")) return;
    setBusy(true);
    try {
      const full = await fetchSkill(name);
      setSelected(full);
      setEditorText(full.raw_md);
      setDirty(false);
    } catch (e: any) {
      flash("Open failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleSave = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      await upsertSkill(selected.name, editorText);
      flash(`Saved ${selected.name}`);
      setDirty(false);
      await refresh();
      const fresh = await fetchSkill(selected.name);
      setSelected(fresh);
      setEditorText(fresh.raw_md);
    } catch (e: any) {
      flash("Save failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete user skill "${name}"? Built-in skills can't be deleted.`)) return;
    setBusy(true);
    try {
      await deleteSkill(name);
      flash(`Deleted ${name}`);
      if (selected?.name === name) {
        setSelected(null);
        setEditorText("");
        setDirty(false);
      }
      await refresh();
    } catch (e: any) {
      flash("Delete failed: " + (e.message || String(e)));
    } finally {
      setBusy(false);
    }
  };

  const handleToggleEnabled = async (sk: SkillMeta) => {
    setBusy(true);
    try {
      await setSkillEnabled(sk.name, !sk.enabled);
      flash(`${sk.name}: ${!sk.enabled ? "enabled" : "disabled"}`);
      await refresh();
      if (selected?.name === sk.name) {
        const fresh = await fetchSkill(sk.name);
        setSelected(fresh);
      }
    } catch (e: any) {
      flash("Toggle failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleReload = async () => {
    setBusy(true);
    try {
      const r = await reloadSkills();
      flash(`Reloaded — ${r.count} skill(s) on disk`);
      await refresh();
    } catch (e: any) {
      flash("Reload failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleCreate = async () => {
    const slug = newName.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "_");
    if (!slug) {
      flash("Name required");
      return;
    }
    setBusy(true);
    try {
      const body = STARTER_MD.replace("name: my_skill", `name: ${slug}`);
      await upsertSkill(slug, body);
      flash(`Created ${slug}`);
      setShowNew(false);
      setNewName("");
      await refresh();
      const fresh = await fetchSkill(slug);
      setSelected(fresh);
      setEditorText(fresh.raw_md);
      setDirty(false);
    } catch (e: any) {
      flash("Create failed: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleInstall = async () => {
    if (!installSource.trim()) {
      flash("Source required");
      return;
    }
    if (
      !confirm(
        "⚠ Installing a skill runs its handler.py inside the agent process " +
        "at next load. Only install from sources you trust. Continue?",
      )
    ) {
      return;
    }
    setBusy(true);
    try {
      const body: SkillInstallRequest = {
        source_type: installType,
        source: installSource.trim(),
        name: installName.trim() || undefined,
        subdir: installSubdir.trim() || undefined,
      };
      const r = await installSkill(body);
      flash(`Installed ${r.name}`);
      setShowInstall(false);
      setInstallSource("");
      setInstallName("");
      setInstallSubdir("");
      await refresh();
    } catch (e: any) {
      flash("Install failed: " + (e.message || String(e)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-3">
        <div className="flex gap-2">
          <button
            onClick={handleReload}
            disabled={busy}
            className="bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Reload from disk
          </button>
          <button
            onClick={() => setShowNew(true)}
            disabled={busy}
            className="bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            + New skill
          </button>
          <button
            onClick={() => setShowInstall(true)}
            disabled={busy}
            className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1.5 text-xs"
          >
            Install from URL / path
          </button>
        </div>
      </div>

      <div className="text-[11px] text-slate-400 mb-3">
        Skills are markdown plugins the agent loads at startup and matches
        against your messages by trigger keywords. Two tiers:
        <span className="text-slate-300"> builtin</span> (ship with the
        engine, refresh on <code>hrant update</code>) and
        <span className="text-emerald-300"> user</span> (live in{" "}
        <code className="font-mono">{userDir || "data_dir/skills"}</code>,
        survive every update). User skills override built-ins of the same
        name.
      </div>

      {/* New skill dialog */}
      {showNew && (
        <div className="bg-slate-800 rounded p-3 mb-3 space-y-2">
          <div className="text-sm font-semibold">New skill</div>
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="name (lowercase, no spaces)"
            className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
          />
          <div className="flex gap-2">
            <button
              onClick={handleCreate}
              disabled={busy}
              className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1 text-xs"
            >
              Create
            </button>
            <button
              onClick={() => { setShowNew(false); setNewName(""); }}
              className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-xs"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Install dialog */}
      {showInstall && (
        <div className="bg-slate-800 rounded p-3 mb-3 space-y-2">
          <div className="text-sm font-semibold">Install skill</div>
          <div className="text-[11px] text-amber-300 bg-amber-900/30 border border-amber-700/50 rounded p-2">
            ⚠ Installing runs the skill's handler.py inside this process.
            Only install from sources you trust.
          </div>
          <div className="flex gap-2">
            {(["git", "zip", "local"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setInstallType(t)}
                className={`rounded px-3 py-1 text-xs ${
                  installType === t ? "bg-sky-700" : "bg-slate-700 hover:bg-slate-600"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <input
            value={installSource}
            onChange={(e) => setInstallSource(e.target.value)}
            placeholder={
              installType === "git"
                ? "https://github.com/user/repo.git"
                : installType === "zip"
                ? "https://example.com/skill.zip"
                : "/absolute/path/to/skill_dir"
            }
            className="w-full bg-slate-900 rounded px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-sky-600 font-mono"
          />
          <div className="grid grid-cols-2 gap-2">
            <input
              value={installName}
              onChange={(e) => setInstallName(e.target.value)}
              placeholder="override name (optional)"
              className="bg-slate-900 rounded px-2 py-1.5 text-xs outline-none focus:ring-1 focus:ring-sky-600 font-mono"
            />
            <input
              value={installSubdir}
              onChange={(e) => setInstallSubdir(e.target.value)}
              placeholder="subdir inside repo/zip (optional)"
              className="bg-slate-900 rounded px-2 py-1.5 text-xs outline-none focus:ring-1 focus:ring-sky-600 font-mono"
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleInstall}
              disabled={busy}
              className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1 text-xs"
            >
              Install
            </button>
            <button
              onClick={() => setShowInstall(false)}
              className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-xs"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="flex gap-3 flex-1 min-h-0">
        {/* Skill list */}
        <div className="w-64 shrink-0 bg-slate-800/60 rounded overflow-y-auto">
          {skills.length === 0 ? (
            <div className="p-3 text-xs text-slate-500">No skills loaded</div>
          ) : (
            skills.map((sk) => (
              <button
                key={sk.name}
                onClick={() => openSkill(sk.name)}
                className={`w-full text-left p-2 border-b border-slate-700/50 text-xs ${
                  selected?.name === sk.name
                    ? "bg-sky-800/60"
                    : "hover:bg-slate-700/50"
                } ${sk.enabled ? "" : "opacity-50"}`}
              >
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="font-semibold flex-1 truncate">{sk.name}</span>
                  <span className={`text-[9px] uppercase rounded px-1 py-0.5 ${SOURCE_COLOUR[sk.source]}`}>
                    {sk.source}
                  </span>
                </div>
                <div className="text-slate-400 truncate text-[10px]">
                  {sk.description}
                </div>
                <div className="text-slate-500 text-[10px]">
                  {sk.triggers.length} triggers · {sk.enabled ? "on" : "off"}
                </div>
              </button>
            ))
          )}
        </div>

        {/* Editor */}
        <div className="flex-1 flex flex-col min-w-0">
          {selected ? (
            <>
              <div className="flex items-center gap-2 mb-2 text-xs">
                <span className={`uppercase rounded px-1.5 py-0.5 text-[10px] ${SOURCE_COLOUR[selected.source]}`}>
                  {selected.source}
                </span>
                <span className="font-mono">{selected.name}</span>
                <span className="text-slate-400 truncate flex-1">{selected.path}</span>
                <button
                  onClick={() => handleToggleEnabled(selected)}
                  disabled={busy}
                  className={`rounded px-2 py-0.5 text-[10px] ${
                    selected.enabled
                      ? "bg-emerald-700 hover:bg-emerald-600"
                      : "bg-slate-600 hover:bg-slate-500"
                  } disabled:opacity-40`}
                >
                  {selected.enabled ? "enabled" : "disabled"}
                </button>
                {selected.source === "user" && (
                  <button
                    onClick={() => handleDelete(selected.name)}
                    disabled={busy}
                    className="bg-rose-800/70 hover:bg-rose-700 disabled:opacity-40 rounded px-2 py-0.5 text-[10px]"
                  >
                    Delete
                  </button>
                )}
              </div>
              <textarea
                value={editorText}
                onChange={(e) => {
                  setEditorText(e.target.value);
                  setDirty(true);
                }}
                className="flex-1 bg-slate-900 rounded p-3 text-xs font-mono outline-none resize-none focus:ring-1 focus:ring-sky-600"
                disabled={busy}
              />
              <div className="flex gap-2 mt-2 text-xs items-center">
                <button
                  onClick={handleSave}
                  disabled={busy || !dirty}
                  className="bg-sky-700 hover:bg-sky-600 disabled:opacity-40 rounded px-3 py-1.5"
                >
                  Save
                </button>
                {selected.source === "builtin" && (
                  <span className="text-amber-300 text-[11px]">
                    Saving will create a USER override (built-in file in repo
                    stays untouched).
                  </span>
                )}
                {dirty && (
                  <span className="text-amber-300 ml-auto">● unsaved</span>
                )}
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-slate-500 text-xs">
              Select a skill on the left, or add a new one.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
