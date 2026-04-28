import IdentityEditor from "./IdentityEditor";

type HistoryVersion = { timestamp: string; path: string; size: number };

type Props = {
  userProfile: string;
  setUserProfile: (v: string) => void;
  dirty: boolean;
  onSave: (file: string, content: string) => void;
  onReload: () => void;
  setDirty: (d: boolean) => void;
  history: HistoryVersion[];
};

export default function UserProfileTab({
  userProfile, setUserProfile, dirty, onSave, onReload, setDirty, history,
}: Props) {
  return (
    <div className="flex flex-col h-full">
      <IdentityEditor
        file="user"
        value={userProfile}
        setter={setUserProfile}
        dirty={dirty}
        onSave={onSave}
        onReload={onReload}
        setDirty={setDirty}
      />
      {history.length > 0 && (
        <details className="mt-3">
          <summary className="text-xs cursor-pointer opacity-60">
            Profile history ({history.length} versions)
          </summary>
          <div className="mt-1 space-y-0.5 text-xs max-h-32 overflow-y-auto">
            {history.map((v, i) => (
              <div key={i} className="flex gap-2 opacity-60">
                <span>{v.timestamp}</span>
                <span>{v.size}b</span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
