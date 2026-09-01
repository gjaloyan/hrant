/** Shared primitives.
 *
 * Before this file the app had 576 hardcoded `bg-slate-N` in six shades
 * with no meaning attached to any of them, so the same kind of element
 * looked different on two screens. These are the small set of shapes the
 * app actually uses; everything new should reach for one of them rather
 * than re-deriving a card out of raw utilities.
 */
import type { ReactNode } from "react";

const cx = (...parts: (string | false | null | undefined)[]) =>
  parts.filter(Boolean).join(" ");

/* ── Structure ─────────────────────────────────────────────────────── */

export function Page({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cx("flex-1 min-w-0 overflow-y-auto", className)}>
      <div className="mx-auto max-w-5xl px-4 py-5 sm:px-6">{children}</div>
    </div>
  );
}

export function Card({
  title,
  subtitle,
  actions,
  children,
  className,
  bodyClass,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section
      className={cx(
        "bg-surface border border-edge rounded-xl2 shadow-card overflow-hidden",
        className,
      )}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 px-4 py-3 border-b border-edge">
          <div className="min-w-0">
            {title && (
              <h3 className="font-semibold text-[15px] leading-snug truncate">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-xs text-ink-dim mt-0.5">{subtitle}</p>
            )}
          </div>
          {actions && (
            <div className="flex items-center gap-2 shrink-0">{actions}</div>
          )}
        </header>
      )}
      <div className={cx("p-4", bodyClass)}>{children}</div>
    </section>
  );
}

/** A labelled band above a group of cards. Gives a screen a spine. */
export function SectionTitle({
  children,
  hint,
  actions,
}: {
  children: ReactNode;
  hint?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex items-end justify-between gap-3 mb-2 mt-1">
      <div>
        <h2 className="text-micro font-semibold uppercase text-ink-dim">
          {children}
        </h2>
        {hint && <p className="text-xs text-ink-faint mt-0.5">{hint}</p>}
      </div>
      {actions}
    </div>
  );
}

/* ── Controls ──────────────────────────────────────────────────────── */

type BtnKind = "primary" | "default" | "ghost" | "danger" | "ok";

const BTN: Record<BtnKind, string> = {
  primary: "bg-accent hover:bg-accent-hover text-white border-transparent",
  default:
    "bg-surface-hover hover:bg-[#26314a] text-ink border-edge-strong",
  ghost:
    "bg-transparent hover:bg-surface-hover text-ink-dim hover:text-ink border-transparent",
  danger: "bg-transparent hover:bg-danger text-danger hover:text-white border-danger/50",
  ok: "bg-transparent hover:bg-ok text-ok hover:text-white border-ok/50",
};

export function Button({
  kind = "default",
  size = "md",
  className,
  children,
  ...rest
}: {
  kind?: BtnKind;
  size?: "sm" | "md";
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 border font-medium whitespace-nowrap",
        size === "sm" ? "text-xs px-2.5 py-1" : "text-sm px-3 py-1.5",
        BTN[kind],
        className,
      )}
    >
      {children}
    </button>
  );
}

/** Label + control + explanation. The explanation is the point: a setting
 *  the user cannot interpret is a setting they will not touch. */
export function Field({
  label,
  hint,
  children,
  className,
}: {
  label: ReactNode;
  hint?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={cx("block", className)}>
      <span className="block text-sm font-medium mb-1">{label}</span>
      {children}
      {hint && <span className="block text-xs text-ink-dim mt-1">{hint}</span>}
    </label>
  );
}

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cx("w-full", props.className)} />;
}

export function Select(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={cx("w-full", props.className)} />;
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  hint?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="flex w-full items-start gap-3 text-left py-2 group"
    >
      <span
        className={cx(
          "mt-0.5 h-5 w-9 shrink-0 rounded-full p-0.5 transition-colors",
          checked ? "bg-accent" : "bg-surface-hover border border-edge-strong",
        )}
      >
        <span
          className={cx(
            "block h-4 w-4 rounded-full bg-white transition-transform",
            checked ? "translate-x-4" : "translate-x-0",
          )}
        />
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-medium">{label}</span>
        {hint && <span className="block text-xs text-ink-dim">{hint}</span>}
      </span>
    </button>
  );
}

/* ── Feedback ──────────────────────────────────────────────────────── */

type Tone = "neutral" | "ok" | "warn" | "danger" | "accent";

const TONE: Record<Tone, string> = {
  neutral: "bg-surface-hover text-ink-dim border-edge-strong",
  ok: "bg-ok/15 text-ok border-ok/30",
  warn: "bg-warn/15 text-warn border-warn/30",
  danger: "bg-danger/15 text-danger border-danger/30",
  accent: "bg-accent-soft text-accent border-accent/30",
};

export function Badge({
  tone = "neutral",
  children,
  title,
}: {
  tone?: Tone;
  children: ReactNode;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cx(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-micro font-medium whitespace-nowrap",
        TONE[tone],
      )}
    >
      {children}
    </span>
  );
}

/** Says what the screen is for and how to fill it — an empty panel that
 *  only says "no data" teaches the user nothing. */
export function EmptyState({
  icon,
  title,
  children,
  action,
}: {
  icon?: ReactNode;
  title: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="text-center py-10 px-4">
      {icon && <div className="text-3xl mb-2 opacity-50">{icon}</div>}
      <p className="font-medium">{title}</p>
      {children && (
        <p className="text-sm text-ink-dim mt-1 max-w-md mx-auto">{children}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={cx(
        "inline-block h-3.5 w-3.5 rounded-full border-2 border-current border-r-transparent animate-spin",
        className,
      )}
    />
  );
}

/** Transient confirmation. Ad-hoc `setMsg` strings were rendered a
 *  different way in almost every panel. */
export function Flash({ text }: { text: string }) {
  if (!text) return null;
  const bad = /error|failed|could not/i.test(text);
  return (
    <div
      role="status"
      className={cx(
        "fixed bottom-16 left-1/2 -translate-x-1/2 z-50 rounded-lg border px-3 py-2 text-sm shadow-pop",
        bad ? TONE.danger : TONE.ok,
      )}
    >
      {text}
    </div>
  );
}

export { cx };
