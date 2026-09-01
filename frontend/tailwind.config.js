/** Semantic tokens.
 *
 * The app had 576 hardcoded `bg-slate-N` across six shades with no meaning
 * attached to any of them — slate-800 was a card here, an input there, and a
 * hover state somewhere else, so the same element looked different on two
 * screens. These names say what a surface IS, not what colour it happens to
 * be, which is what makes a change land everywhere at once.
 *
 * The values sit deliberately close to the slate family the codebase already
 * uses, so panels not yet ported blend in instead of clashing.
 */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "var(--canvas)",
        surface: {
          DEFAULT: "var(--surface)",
          raised: "var(--surface-2)",
          hover: "var(--surface-3)",
        },
        edge: {
          DEFAULT: "var(--border)",
          strong: "var(--border-strong)",
        },
        ink: {
          DEFAULT: "var(--text)",
          dim: "var(--text-dim)",
          faint: "var(--text-faint)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          hover: "var(--accent-hover)",
          soft: "var(--accent-soft)",
        },
        ok: "var(--ok)",
        warn: "var(--warn)",
        danger: "var(--danger)",
      },
      borderRadius: { xl2: "0.875rem" },
      fontSize: {
        // A real scale, so headings stop being "text-sm but bold".
        micro: ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.04em" }],
      },
      boxShadow: {
        card: "0 1px 2px rgba(0,0,0,.35)",
        pop: "0 8px 28px rgba(0,0,0,.45)",
      },
    },
  },
  plugins: [],
};
