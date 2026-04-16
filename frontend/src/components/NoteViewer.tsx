import { useEffect, useState } from "react";
import { deleteNote, fetchNote } from "../api";

export default function NoteViewer({
  topic,
  onClose,
}: {
  topic: string | null;
  onClose: () => void;
}) {
  const [note, setNote] = useState<any>(null);

  useEffect(() => {
    if (!topic) {
      setNote(null);
      return;
    }
    fetchNote(topic).then(setNote);
  }, [topic]);

  if (!topic) return null;

  const handleDelete = async () => {
    if (!confirm(`Delete note "${topic}"?`)) return;
    await deleteNote(topic);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="bg-slate-900 rounded-lg p-5 max-w-2xl max-h-[80vh] overflow-y-auto w-full mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        {!note ? (
          <p>loading...</p>
        ) : (
          <>
            <div className="flex justify-between items-start mb-2">
              <h2 className="text-xl font-bold">{note.frontmatter.topic}</h2>
              <div className="flex gap-2">
                <button
                  onClick={handleDelete}
                  className="text-rose-400 hover:text-rose-300 text-sm px-2"
                >
                  delete
                </button>
                <button onClick={onClose} className="opacity-60 hover:opacity-100 text-lg">
                  x
                </button>
              </div>
            </div>
            <div className="text-xs opacity-60 mb-3 space-y-0.5">
              <div>category: {note.frontmatter.category}</div>
              <div>source: {note.frontmatter.source}</div>
              <div>confidence: {note.frontmatter.confidence}</div>
              <div>access count: {note.frontmatter.access_count}</div>
              {note.frontmatter.keywords?.length > 0 && (
                <div>keywords: {note.frontmatter.keywords.join(", ")}</div>
              )}
            </div>
            <pre className="whitespace-pre-wrap text-sm bg-slate-950 rounded p-3">
              {note.body}
            </pre>
          </>
        )}
      </div>
    </div>
  );
}
