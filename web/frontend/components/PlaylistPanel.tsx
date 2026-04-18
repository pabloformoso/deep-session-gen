"use client";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import type { Track } from "@/lib/types";

interface TrackRowProps {
  track: Track;
  position: number;
}

function TrackRow({ track, position }: TrackRowProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: track.id || `track-${position}`,
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const dur = track.duration_sec
    ? `${Math.floor(track.duration_sec / 60)}:${String(Math.round(track.duration_sec % 60)).padStart(2, "0")}`
    : "?:??";

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="flex items-center gap-3 py-2 px-3 rounded hover:bg-[#1e1e2e]/50 group cursor-default"
    >
      <span
        {...attributes}
        {...listeners}
        className="text-muted cursor-grab active:cursor-grabbing select-none"
        title="Drag to reorder"
      >
        ⠿
      </span>

      <span className="text-muted text-xs w-5 text-right flex-shrink-0">{position}</span>

      <div className="flex-1 min-w-0">
        <p className="text-sm text-[#e2e2ff] truncate">{track.display_name}</p>
        <p className="text-xs text-muted">{track.genre}</p>
      </div>

      <div className="flex items-center gap-2 flex-shrink-0 text-xs">
        {track.camelot_key && (
          <span className="text-neon font-bold">{track.camelot_key}</span>
        )}
        {track.bpm && (
          <span className="text-muted">{Math.round(track.bpm)} BPM</span>
        )}
        <span className="text-muted">{dur}</span>
      </div>
    </div>
  );
}

interface PlaylistPanelProps {
  tracks: Track[];
  onReorder?: (newOrder: Track[]) => void;
}

export default function PlaylistPanel({ tracks, onReorder }: PlaylistPanelProps) {
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over || active.id === over.id || !onReorder) return;
    const oldIndex = tracks.findIndex(t => (t.id || `track-${tracks.indexOf(t)}`) === active.id);
    const newIndex = tracks.findIndex(t => (t.id || `track-${tracks.indexOf(t)}`) === over.id);
    onReorder(arrayMove(tracks, oldIndex, newIndex));
  }

  if (tracks.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-muted text-xs">No playlist yet</p>
      </div>
    );
  }

  const totalSec = tracks.reduce((sum, t) => sum + (t.duration_sec ?? 0), 0);
  const totalMin = Math.round(totalSec / 60);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border">
        <span className="text-xs text-muted uppercase tracking-widest">Playlist</span>
        <span className="text-xs text-muted">{tracks.length} tracks · {totalMin}min</span>
      </div>

      <div className="flex-1 overflow-y-auto">
        <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
          <SortableContext
            items={tracks.map((t, i) => t.id || `track-${i}`)}
            strategy={verticalListSortingStrategy}
          >
            {tracks.map((track, i) => (
              <TrackRow key={track.id || i} track={track} position={i + 1} />
            ))}
          </SortableContext>
        </DndContext>
      </div>
    </div>
  );
}
