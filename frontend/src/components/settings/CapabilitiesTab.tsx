type Props = {
  capabilities: string;
  onRefresh: () => void;
};

export default function CapabilitiesTab({ capabilities, onRefresh }: Props) {
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <button
          onClick={onRefresh}
          className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-xs"
        >
          Refresh
        </button>
      </div>
      <pre className="whitespace-pre-wrap text-sm bg-slate-950 rounded p-4">
        {capabilities || "(loading...)"}
      </pre>
    </div>
  );
}
