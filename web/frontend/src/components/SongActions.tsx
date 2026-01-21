import { useState } from 'react'
import { Download, Scissors, ArrowRight, ArrowLeft, Clock } from 'lucide-react'
import { formatDuration } from '../lib/utils'

interface SongActionsProps {
  songId: string
  songName: string
  durationMs: number
  downloadUrl: string
  selectedTimeMs: number | null
  onExtend: (params: {
    extend_from_ms: number
    extend_duration_ms: number
    direction: 'before' | 'after'
    prompt?: string
  }) => Promise<void>
  onCrop: (startMs: number, endMs: number) => Promise<void>
  authToken: string | null
}

export function SongActions({
  songId: _songId,
  songName,
  durationMs,
  downloadUrl,
  selectedTimeMs,
  onExtend,
  onCrop,
  authToken,
}: SongActionsProps) {
  const [showExtend, setShowExtend] = useState(false)
  const [showCrop, setShowCrop] = useState(false)
  const [extendDuration, setExtendDuration] = useState(30)
  const [extendDirection, setExtendDirection] = useState<'before' | 'after'>('after')
  const [extendPrompt, setExtendPrompt] = useState('')
  const [cropStart, setCropStart] = useState(0)
  const [cropEnd, setCropEnd] = useState(durationMs)
  const [loading, setLoading] = useState(false)

  const handleDownload = async () => {
    if (!authToken) return
    
    try {
      const response = await fetch(downloadUrl, {
        headers: { Authorization: `Bearer ${authToken}` },
      })
      
      if (!response.ok) throw new Error('Download failed')
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${songName}.wav`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Download error:', error)
    }
  }

  const handleExtend = async () => {
    if (selectedTimeMs === null) return
    
    setLoading(true)
    try {
      await onExtend({
        extend_from_ms: selectedTimeMs,
        extend_duration_ms: extendDuration * 1000,
        direction: extendDirection,
        prompt: extendPrompt || undefined,
      })
      setShowExtend(false)
    } finally {
      setLoading(false)
    }
  }

  const handleCrop = async () => {
    setLoading(true)
    try {
      await onCrop(cropStart, cropEnd)
      setShowCrop(false)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-studio-panel rounded-lg border border-studio-border p-4 space-y-4">
      <h3 className="font-medium text-studio-text">Song Actions</h3>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={handleDownload}
          className="flex items-center gap-2 px-3 py-2 bg-studio-bg border border-studio-border rounded-lg text-studio-text hover:border-studio-accent transition-colors"
        >
          <Download className="w-4 h-4" />
          Download
        </button>

        <button
          onClick={() => {
            setShowExtend(!showExtend)
            setShowCrop(false)
          }}
          className={`flex items-center gap-2 px-3 py-2 border rounded-lg transition-colors ${
            showExtend
              ? 'bg-studio-accent border-studio-accent text-white'
              : 'bg-studio-bg border-studio-border text-studio-text hover:border-studio-accent'
          }`}
        >
          <ArrowRight className="w-4 h-4" />
          Extend
        </button>

        <button
          onClick={() => {
            setShowCrop(!showCrop)
            setShowExtend(false)
          }}
          className={`flex items-center gap-2 px-3 py-2 border rounded-lg transition-colors ${
            showCrop
              ? 'bg-studio-accent border-studio-accent text-white'
              : 'bg-studio-bg border-studio-border text-studio-text hover:border-studio-accent'
          }`}
        >
          <Scissors className="w-4 h-4" />
          Crop
        </button>
      </div>

      {showExtend && (
        <div className="p-3 bg-studio-bg rounded-lg space-y-3">
          <h4 className="text-sm font-medium text-studio-text">Extend Song</h4>

          {selectedTimeMs === null ? (
            <p className="text-sm text-yellow-400">
              Shift+Click on the waveform to select where to extend from
            </p>
          ) : (
            <>
              <p className="text-sm text-studio-muted">
                Extend from: <span className="text-studio-accent">{formatDuration(selectedTimeMs)}</span>
              </p>

              <div>
                <label className="block text-sm text-studio-muted mb-1">Direction</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => setExtendDirection('before')}
                    className={`flex-1 flex items-center justify-center gap-1 px-3 py-2 rounded-lg border transition-colors ${
                      extendDirection === 'before'
                        ? 'bg-studio-accent border-studio-accent text-white'
                        : 'bg-studio-panel border-studio-border text-studio-muted hover:text-studio-text'
                    }`}
                  >
                    <ArrowLeft className="w-4 h-4" />
                    Before
                  </button>
                  <button
                    onClick={() => setExtendDirection('after')}
                    className={`flex-1 flex items-center justify-center gap-1 px-3 py-2 rounded-lg border transition-colors ${
                      extendDirection === 'after'
                        ? 'bg-studio-accent border-studio-accent text-white'
                        : 'bg-studio-panel border-studio-border text-studio-muted hover:text-studio-text'
                    }`}
                  >
                    After
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm text-studio-muted mb-1">
                  <Clock className="w-3 h-3 inline mr-1" />
                  Extension Duration: {extendDuration}s
                </label>
                <input
                  type="range"
                  min={10}
                  max={60}
                  step={5}
                  value={extendDuration}
                  onChange={(e) => setExtendDuration(Number(e.target.value))}
                  className="w-full accent-studio-accent"
                />
              </div>

              <div>
                <label className="block text-sm text-studio-muted mb-1">
                  Prompt (optional)
                </label>
                <input
                  type="text"
                  value={extendPrompt}
                  onChange={(e) => setExtendPrompt(e.target.value)}
                  placeholder="Leave empty to continue with original style"
                  className="w-full bg-studio-panel border border-studio-border rounded-lg p-2 text-studio-text placeholder-studio-muted focus:outline-none focus:border-studio-accent"
                />
              </div>

              <button
                onClick={handleExtend}
                disabled={loading}
                className="w-full bg-studio-accent hover:bg-blue-600 text-white font-medium py-2 rounded-lg transition-colors disabled:opacity-50"
              >
                {loading ? 'Extending...' : 'Extend Song (5 credits)'}
              </button>
            </>
          )}
        </div>
      )}

      {showCrop && (
        <div className="p-3 bg-studio-bg rounded-lg space-y-3">
          <h4 className="text-sm font-medium text-studio-text">Crop Song</h4>

          <div>
            <label className="block text-sm text-studio-muted mb-1">
              Start: {formatDuration(cropStart)}
            </label>
            <input
              type="range"
              min={0}
              max={durationMs - 1000}
              step={100}
              value={cropStart}
              onChange={(e) => {
                const val = Number(e.target.value)
                setCropStart(val)
                if (val >= cropEnd) setCropEnd(val + 1000)
              }}
              className="w-full accent-studio-accent"
            />
          </div>

          <div>
            <label className="block text-sm text-studio-muted mb-1">
              End: {formatDuration(cropEnd)}
            </label>
            <input
              type="range"
              min={1000}
              max={durationMs}
              step={100}
              value={cropEnd}
              onChange={(e) => {
                const val = Number(e.target.value)
                setCropEnd(val)
                if (val <= cropStart) setCropStart(val - 1000)
              }}
              className="w-full accent-studio-accent"
            />
          </div>

          <p className="text-sm text-studio-muted">
            New duration: <span className="text-studio-accent">{formatDuration(cropEnd - cropStart)}</span>
          </p>

          <button
            onClick={handleCrop}
            disabled={loading || cropEnd <= cropStart}
            className="w-full bg-studio-accent hover:bg-blue-600 text-white font-medium py-2 rounded-lg transition-colors disabled:opacity-50"
          >
            {loading ? 'Cropping...' : 'Crop Song'}
          </button>
        </div>
      )}
    </div>
  )
}
