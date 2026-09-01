/** Which model handles which kind of work.
 *
 * The backend has routed per task type since 2026-06-11 and nothing in the
 * UI could reach it, so on prod it sat `{"enabled": false, "routing": {}}`
 * — never configured, because there was no way to configure it.
 *
 * That matters more than most settings. Every task turn fires one to three
 * CLASSIFICATION calls plus keyword extraction and memory work, and with
 * routing off all of them go to the expensive pinned model. That is a large
 * part of why the input:output ratio sits above 20:1.
 *
 * The screen is built around the decision, not the config file: the cheap,
 * mechanical task types are listed first and described in plain terms, and
 * anything left unset simply uses the main model.
 */
import { useEffect, useState } from "react";
import {
  fetchCascade,
  fetchModelRouting,
  fetchProviders,
  putModelRouting,
  saveCascade,
  type CascadeConfig,
  type ModelRoutingEntry,
  type ProviderConfig,
} from "../../api";
import { Badge, Button, Card, Spinner, Toggle, cx } from "../../ui";

/** The routable task types, in the order worth thinking about them.
 *  `cheap: true` marks the mechanical ones — short prompts, structured
 *  output, no judgement — where a small model is genuinely enough. */
const TASKS: { id: string; label: string; hint: string; cheap: boolean }[] = [
  { id: "classification", label: "Classification", hint: "Deciding what a message is. Fires 1–3 times per turn.", cheap: true },
  { id: "keyword_extraction", label: "Keyword extraction", hint: "Pulling search terms out of text.", cheap: true },
  { id: "note_search", label: "Note search", hint: "Picking which saved notes are relevant.", cheap: true },
  { id: "simple_lookup", label: "Simple lookup", hint: "Short factual answers from what it already has.", cheap: true },
  { id: "quick_answer", label: "Quick answer", hint: "Short replies that need no research.", cheap: true },
  { id: "note_creation", label: "Note writing", hint: "Turning a finding into a saved note.", cheap: true },
  { id: "task_analysis", label: "Task analysis", hint: "Working out what a request actually needs.", cheap: false },
  { id: "verification", label: "Verification", hint: "Checking its own answer. A weak model here is dangerous.", cheap: false },
  { id: "self_critic", label: "Self-critique", hint: "The revise pass after verification.", cheap: false },
  { id: "skill_reflection", label: "Skill reflection", hint: "Deciding whether a turn is worth saving as a skill.", cheap: false },
  { id: "learning", label: "Learning", hint: "Research and study passes.", cheap: false },
  { id: "complex_solving", label: "Complex solving", hint: "The hard work. Leave this on the main model.", cheap: false },
];

export default function ModelRoutingTab({
  flash,
}: {
  flash: (msg: string) => void;
}) {
  const [enabled, setEnabled] = useState(false);
  const [routing, setRouting] = useState<Record<string, ModelRoutingEntry>>({});
  const [providers, setProviders] = useState<ProviderConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [showAll, setShowAll] = useState(false);
  // The cascade lived on the Fine-Tune data screen, between an example
  // browser and a JSONL export — a different question entirely. It is the
  // same decision as the table below (which model does what), so it lives
  // with it, and it picks a model from a list instead of asking the reader
  // to type a provider id from memory.
  const [cascade, setCascade] = useState<CascadeConfig | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [cfg, pr, cas] = await Promise.all([
          fetchModelRouting(),
          fetchProviders(),
          fetchCascade().catch(() => null),
        ]);
        setEnabled(!!cfg.enabled);
        setRouting(cfg.routing || {});
        setProviders((pr as any).providers || []);
        if (cas) setCascade(cas as CascadeConfig);
      } catch (e: any) {
        flash("Error: " + e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  /** Every (provider, model) pair the box can actually reach. */
  const options = providers.flatMap((p: any) =>
    (p.models || []).map((m: string) => ({
      key: `${p.id}::${m}`,
      provider_id: p.id,
      model: m,
      label: `${p.name || p.id} · ${m}`,
    })),
  );

  const set = (task: string, key: string) => {
    setDirty(true);
    setRouting((prev) => {
      const next = { ...prev };
      if (!key) delete next[task];
      else {
        const o = options.find((x) => x.key === key);
        if (o) next[task] = { provider_id: o.provider_id, model: o.model };
      }
      return next;
    });
  };

  const save = async () => {
    setSaving(true);
    try {
      await putModelRouting(enabled, routing);
      setDirty(false);
      flash("Routing saved");
    } catch (e: any) {
      flash("Error: " + e.message);
    } finally {
      setSaving(false);
    }
  };

  if (loading)
    return (
      <div className="flex items-center gap-2 text-sm text-ink-dim">
        <Spinner /> Loading…
      </div>
    );

  const rows = showAll ? TASKS : TASKS.filter((t) => t.cheap);
  const assigned = Object.keys(routing).length;

  return (
    <div className="max-w-3xl space-y-4">
      <Card>
        <Toggle
          checked={enabled}
          onChange={(v) => {
            setEnabled(v);
            setDirty(true);
          }}
          label="Send mechanical work to a cheaper model"
          hint="When off, every call goes to the main model, including the one to three classification calls fired on each turn."
        />
        {enabled && assigned === 0 && (
          <p className="mt-2 text-xs text-warn">
            Nothing is routed yet, so this changes nothing. Assign at least
            one task type below.
          </p>
        )}
      </Card>

      <Card
        title="Task types"
        subtitle="Anything left on “Main model” behaves exactly as it does today."
        actions={
          <Button kind="ghost" size="sm" onClick={() => setShowAll((v) => !v)}>
            {showAll ? "Show cheap tasks only" : "Show all task types"}
          </Button>
        }
      >
        {options.length === 0 && (
          <p className="mb-3 text-sm text-warn">
            No models available. Add a provider under Settings → Providers
            first.
          </p>
        )}
        <ul className="divide-y divide-edge">
          {rows.map((t) => {
            const cur = routing[t.id];
            const key = cur ? `${cur.provider_id}::${cur.model}` : "";
            return (
              <li key={t.id} className="flex flex-wrap items-center gap-3 py-2.5">
                <div className="min-w-0 flex-1">
                  <p className="flex items-center gap-2 text-sm font-medium">
                    {t.label}
                    {t.cheap && <Badge tone="ok">cheap</Badge>}
                    {!t.cheap && t.id === "verification" && (
                      <Badge tone="warn">be careful</Badge>
                    )}
                  </p>
                  <p className="text-xs text-ink-dim">{t.hint}</p>
                </div>
                <select
                  value={key}
                  disabled={!enabled}
                  onChange={(e) => set(t.id, e.target.value)}
                  className={cx("w-64 text-sm", !enabled && "opacity-50")}
                  aria-label={`Model for ${t.label}`}
                >
                  <option value="">Main model</option>
                  {options.map((o) => (
                    <option key={o.key} value={o.key}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </li>
            );
          })}
        </ul>
      </Card>

      {cascade && (
        <Card
          title="Try a small model first"
          subtitle="A cheap model answers, a strong one checks it, and only weak answers are redone by the main model. Separate from the table above, which never re-runs anything."
        >
          <Toggle
            checked={cascade.enabled}
            onChange={(v) => {
              const next = { ...cascade, enabled: v };
              setCascade(next);
              saveCascade(next).then(
                () => flash(v ? "Cascade on" : "Cascade off"),
                (e) => flash("Error: " + e.message),
              );
            }}
            label="Answer with a small model first"
            hint="Costs a verifier call on every turn, and saves the main model's price on the ones that pass."
          />

          {cascade.enabled && (
            <div className="mt-3 space-y-3 border-t border-edge pt-3">
              <label className="block">
                <span className="mb-1 block text-sm font-medium">
                  Small model
                </span>
                <select
                  value={`${cascade.provider_id}::${cascade.model}`}
                  onChange={(e) => {
                    const o = options.find((x) => x.key === e.target.value);
                    if (!o) return;
                    const next = {
                      ...cascade,
                      provider_id: o.provider_id,
                      model: o.model,
                    };
                    setCascade(next);
                    saveCascade(next).then(() => flash("Saved"));
                  }}
                  className="w-full max-w-md text-sm"
                >
                  <option value={`${cascade.provider_id}::${cascade.model}`}>
                    {cascade.provider_id
                      ? `${cascade.provider_id} · ${cascade.model}`
                      : "(none chosen)"}
                  </option>
                  {options
                    .filter(
                      (o) => o.key !== `${cascade.provider_id}::${cascade.model}`,
                    )
                    .map((o) => (
                      <option key={o.key} value={o.key}>
                        {o.label}
                      </option>
                    ))}
                </select>
              </label>

              <label className="block">
                <span className="mb-1 block text-sm font-medium">
                  Escalate below {cascade.confidence_gate}% confidence
                </span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={cascade.confidence_gate}
                  onChange={(e) =>
                    setCascade({
                      ...cascade,
                      confidence_gate: Number(e.target.value),
                    })
                  }
                  onMouseUp={() => saveCascade(cascade).then(() => flash("Saved"))}
                  className="w-full max-w-md"
                />
                <span className="block text-xs text-ink-dim">
                  Higher sends more work to the main model. Lower saves more
                  and risks shipping a weaker answer.
                </span>
              </label>
            </div>
          )}
        </Card>
      )}

      <div className="flex items-center gap-3">
        <Button kind="primary" onClick={save} disabled={!dirty || saving}>
          {saving ? <Spinner /> : "Save routing"}
        </Button>
        {dirty && <span className="text-xs text-warn">Unsaved changes</span>}
      </div>
    </div>
  );
}
