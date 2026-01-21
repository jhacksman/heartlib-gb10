import { Music, Clock, Loader2, CheckCircle, XCircle, Trash2, ListOrdered } from 'lucide-react'
import { formatDuration } from '../lib/utils'

interface Song {
  id: string
  name: string
  prompt: string
  tags: string
  duration_ms: number
  status: string
  progress: number
  message: string
  output_url: string | null
  created_at: string
}

interface SongListProps {
  songs: Song[]
  selectedSongId: string | null
  onSelectSong: (song: Song) => void
  onDeleteSong: (songId: string) => void
}

export function SongList({ songs, selectedSongId, onSelectSong, onDeleteSong }: SongListProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <ListOrdered className="w-4 h-4 text-yellow-400" />
      case 'processing':
        return <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-400" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />
      default:
        return null
    }
  }

  const getStatusText = (song: Song) => {
    switch (song.status) {
      case 'pending':
        return 'Queued'
      case 'processing':
        return song.message || 'Generating...'
      case 'completed':
        return 'Ready'
      case 'failed':
        return song.message || 'Failed'
      default:
        return song.status
    }
  }

  if (songs.length === 0) {
    return (
      <div className="bg-studio-panel rounded-lg border border-studio-border p-6 text-center">
        <Music className="w-12 h-12 text-studio-muted mx-auto mb-3" />
        <p className="text-studio-muted">No songs yet</p>
        <p className="text-sm text-studio-muted mt-1">Create your first song to get started</p>
      </div>
    )
  }

  return (
    <div className="bg-studio-panel rounded-lg border border-studio-border overflow-hidden">
      <div className="p-3 border-b border-studio-border">
        <h3 className="font-medium text-studio-text">Your Songs ({songs.length})</h3>
      </div>
      
      <div className="max-h-96 overflow-y-auto">
        {songs.map((song) => (
          <div
            key={song.id}
            onClick={() => onSelectSong(song)}
            className={`p-3 border-b border-studio-border cursor-pointer transition-colors ${
              selectedSongId === song.id
                ? 'bg-studio-accent/20'
                : 'hover:bg-studio-bg'
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  {getStatusIcon(song.status)}
                  <span className="font-medium text-studio-text truncate">
                    {song.name}
                  </span>
                </div>
                
                <p className="text-sm text-studio-muted truncate mt-1">
                  {song.prompt}
                </p>
                
                <div className="flex items-center gap-3 mt-2 text-xs text-studio-muted">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {formatDuration(song.duration_ms)}
                  </span>
                  <span>{getStatusText(song)}</span>
                </div>
                
                {song.status === 'processing' && (
                  <div className="mt-2">
                    <div className="w-full bg-studio-bg rounded-full h-1.5 overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-300 ease-out"
                        style={{ width: `${Math.round(song.progress * 100)}%` }}
                      />
                    </div>
                    <div className="text-xs text-studio-muted mt-1 text-right">
                      {Math.round(song.progress * 100)}%
                    </div>
                  </div>
                )}
                
                {song.tags && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {song.tags.split(',').slice(0, 3).map((tag, i) => (
                      <span
                        key={i}
                        className="px-1.5 py-0.5 text-xs bg-studio-bg rounded text-studio-muted"
                      >
                        {tag.trim()}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onDeleteSong(song.id)
                }}
                className="p-1.5 text-studio-muted hover:text-red-400 transition-colors"
                title="Delete song"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
