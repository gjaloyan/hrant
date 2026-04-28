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
      <div className="flex justify-between items-center mb-2">
        <h3 className="font-bold">{file}.md</h3>
        <div className="flex gap-2">
          {dirty && <span className="text-xs text-amber-400">unsaved changes</span>}
          <button
            onClick={() => onSave(file, value)}
            disabled={!dirty}
            className="bg-emerald-700 hover:bg-emerald-600 disabled:opacity-40 rounded px-3 py-1 text-xs"
          >
            Save
          </button>
          <button
            onClick={onReload}
            className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-xs"
          >
            Reload
          </button>
        </div>
      </div>
      <textarea
        className="flex-1 bg-slate-950 rounded p-3 text-sm font-mono resize-none outline-none focus:ring-1 focus:ring-sky-600"
        value={value}
        onChange={(e) => {
          setter(e.target.value);
          setDirty(true);
        }}
      />
    </div>
  );
}
