import { Badge, Button } from "../../ui";
type Props = {
  file: string;
  value: string;
  setter: (v: string) => void;
  dirty: boolean;
  onSave: (file: string, content: string) => void;
  onReload: () => void;
  setDirty: (d: boolean) => void;
};

export default function IdentityEditor({ file, value, setter, dirty, onSave, onReload, setDirty }: Props) {
  return (
    <div className="flex flex-col h-full">
      {/* Save was olive and Reload was grey, so neither read as the main
          action and "unsaved changes" was a small amber note beside them.
          Save is now the primary, lit only when there is something to save;
          the unsaved state is a badge rather than loose text. */}
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <h3 className="truncate font-mono text-sm text-ink-dim">{file}.md</h3>
          {dirty && <Badge tone="warn">unsaved</Badge>}
        </div>
        <div className="flex shrink-0 gap-2">
          <Button kind="ghost" size="sm" onClick={onReload}
                  title="Discard edits and reload from disk">
            Revert
          </Button>
          <Button
            kind="primary"
            size="sm"
            onClick={() => onSave(file, value)}
            disabled={!dirty}
          >
            Save
          </Button>
        </div>
      </div>
      <textarea
        className="flex-1 resize-none rounded-xl2 border border-edge bg-canvas p-3 font-mono text-sm"
        value={value}
        onChange={(e) => {
          setter(e.target.value);
          setDirty(true);
        }}
      />
    </div>
  );
}
