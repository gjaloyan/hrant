/** Who a speaker id belongs to.
 *
 * Panels were rendering raw ids — `telegram:1358056500` in the Sessions
 * list, `webui:default` in reminders — while `/api/roles` has carried a
 * label ("Gor", "Tigran", "Wife") and a role for every one of them all
 * along. An id nobody can read makes a list unscannable and forces the
 * reader to hold a lookup table in their head.
 *
 * Fetched once per page load and shared, because several panels need the
 * same map and it changes about as often as the household does.
 */
import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { fetchRoles } from "../api";
import { Badge } from "./index";

export type SpeakerInfo = { label: string; role: string; isOwner: boolean };

type Map = Record<string, SpeakerInfo>;

const Ctx = createContext<Map>({});

export function SpeakerProvider({ children }: { children: ReactNode }) {
  const [map, setMap] = useState<Map>({});

  useEffect(() => {
    let dead = false;
    fetchRoles()
      .then((r: any) => {
        if (dead) return;
        const owners: string[] = r.owner_speaker_ids || [];
        const out: Map = {};
        for (const [id, entry] of Object.entries<any>(r.speakers || {})) {
          out[id] = {
            label: entry?.label || "",
            role: entry?.role || "guest",
            isOwner: entry?.role === "owner" || owners.includes(id),
          };
        }
        setMap(out);
      })
      .catch(() => {
        /* names are a nicety; the ids still render */
      });
    return () => {
      dead = true;
    };
  }, []);

  return <Ctx.Provider value={map}>{children}</Ctx.Provider>;
}

export function useSpeakers(): Map {
  return useContext(Ctx);
}

/** A readable channel prefix. `telegram:848…` is noise; "Telegram" is not. */
export function channelOf(id: string): string {
  const head = (id || "").split(":")[0];
  return (
    { telegram: "Telegram", webui: "Web", cli: "CLI", api: "API",
      voice: "Voice", audit: "Audit" }[head] || head || "—"
  );
}

/** Name if we know it, else a short form of the id — never the raw string
 *  with a 12-digit account number in it. */
export function nameOf(id: string, map: Map): string {
  const known = map[id]?.label;
  if (known) return known;
  const [head, ...rest] = (id || "").split(":");
  const tail = rest.join(":");
  if (!tail) return id || "—";
  return `${channelOf(head)} ${tail.length > 8 ? "…" + tail.slice(-4) : tail}`;
}

/** Name plus role, for a list row. The full id stays in the tooltip so
 *  nothing is actually hidden. */
export function Speaker({ id, showRole = false }: { id: string; showRole?: boolean }) {
  const map = useSpeakers();
  const info = map[id];
  return (
    <span className="inline-flex items-center gap-1.5" title={id}>
      <span className="truncate">{nameOf(id, map)}</span>
      {showRole && info?.role && (
        <Badge tone={info.isOwner ? "accent" : info.role === "trusted" ? "ok" : "neutral"}>
          {info.role}
        </Badge>
      )}
    </span>
  );
}
