import { ConversationTurn } from "../../api";

type Props = {
  conversation: ConversationTurn[];
  convCount: number;
  onRefresh: () => void;
  onClear: () => void;
};

export default function ConversationTab({ conversation, convCount, onRefresh, onClear }: Props) {
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <h3 className="font-bold">Conversation History ({convCount} turns)</h3>
        <div className="flex gap-2">
          <button
            onClick={onRefresh}
            className="bg-slate-700 hover:bg-slate-600 rounded px-3 py-1 text-xs"
          >
            Refresh
          </button>
          <button
            onClick={onClear}
            className="bg-rose-700 hover:bg-rose-600 rounded px-3 py-1 text-xs"
          >
            Clear all
          </button>
        </div>
      </div>
      {conversation.length === 0 && (
        <div className="opacity-50 text-sm">(no conversation history)</div>
      )}
      {conversation.map((turn, i) => (
        <div key={i} className="bg-slate-800 rounded p-3 text-sm space-y-1">
          <div className="flex gap-2 text-xs opacity-60">
            <span>{turn.ts}</span>
            <span className="bg-slate-700 px-1 rounded">{turn.intent}</span>
            {turn.confidence ? <span>conf: {turn.confidence}%</span> : null}
            {turn.topics?.length ? <span>topics: {turn.topics.join(", ")}</span> : null}
          </div>
          <div>
            <span className="text-sky-400 font-semibold">User: </span>
            {turn.user}
          </div>
          <div>
            <span className="text-emerald-400 font-semibold">Agent: </span>
            {turn.answer}
          </div>
        </div>
      ))}
    </div>
  );
}
