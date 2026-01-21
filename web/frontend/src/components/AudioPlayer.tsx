import { useState, useRef, useEffect } from 'react'
import { Play, Pause, SkipBack, SkipForward, Volume2, VolumeX } from 'lucide-react'
import { formatDuration } from '../lib/utils'

interface AudioPlayerProps {
  src: string | null
  duration_ms: number
  onTimeSelect?: (timeMs: number) => void
  selectedTime?: number | null
}

export function AudioPlayer({ src, duration_ms, onTimeSelect, selectedTime }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(duration_ms / 1000)
  const [volume, setVolume] = useState(1)
  const [isMuted, setIsMuted] = useState(false)

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const handleTimeUpdate = () => setCurrentTime(audio.currentTime)
    const handleLoadedMetadata = () => setDuration(audio.duration || duration_ms / 1000)
    const handleEnded = () => setIsPlaying(false)

    audio.addEventListener('timeupdate', handleTimeUpdate)
    audio.addEventListener('loadedmetadata', handleLoadedMetadata)
    audio.addEventListener('ended', handleEnded)

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate)
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata)
      audio.removeEventListener('ended', handleEnded)
    }
  }, [duration_ms])

  useEffect(() => {
    drawWaveform()
  }, [currentTime, selectedTime, duration])

  const drawWaveform = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.fillStyle = '#22262e'
    ctx.fillRect(0, 0, width, height)

    // Draw waveform bars (simulated)
    const barCount = 100
    const barWidth = width / barCount - 1
    const barGap = 1

    for (let i = 0; i < barCount; i++) {
      const x = i * (barWidth + barGap)
      
      // Generate pseudo-random height based on position
      const seed = Math.sin(i * 0.5) * Math.cos(i * 0.3) + Math.sin(i * 0.1)
      const barHeight = Math.abs(seed) * 0.6 * height + height * 0.1

      const progress = currentTime / duration
      const isPlayed = i / barCount <= progress

      // Check if this bar is at the selected time
      const isSelected = selectedTime !== null && selectedTime !== undefined && 
        Math.abs(i / barCount - selectedTime / (duration * 1000)) < 0.02

      if (isSelected) {
        ctx.fillStyle = '#ef4444' // Red for selected
      } else if (isPlayed) {
        ctx.fillStyle = '#3b82f6' // Blue for played
      } else {
        ctx.fillStyle = '#4b5563' // Gray for unplayed
      }

      const y = (height - barHeight) / 2
      ctx.fillRect(x, y, barWidth, barHeight)
    }

    // Draw playhead
    const playheadX = (currentTime / duration) * width
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(playheadX - 1, 0, 2, height)

    // Draw selected time marker if present
    if (selectedTime !== null && selectedTime !== undefined) {
      const markerX = (selectedTime / (duration * 1000)) * width
      ctx.fillStyle = '#ef4444'
      ctx.fillRect(markerX - 1, 0, 2, height)
      
      // Draw triangle marker at top
      ctx.beginPath()
      ctx.moveTo(markerX - 6, 0)
      ctx.lineTo(markerX + 6, 0)
      ctx.lineTo(markerX, 10)
      ctx.closePath()
      ctx.fill()
    }
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const audio = audioRef.current
    if (!canvas || !audio) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const clickPercent = x / rect.width
    const newTime = clickPercent * duration

    if (e.shiftKey && onTimeSelect) {
      // Shift+click to select time for extend/crop
      onTimeSelect(newTime * 1000)
    } else {
      // Regular click to seek
      audio.currentTime = newTime
      setCurrentTime(newTime)
    }
  }

  const togglePlay = () => {
    const audio = audioRef.current
    if (!audio || !src) return

    if (isPlaying) {
      audio.pause()
    } else {
      audio.play()
    }
    setIsPlaying(!isPlaying)
  }

  const toggleMute = () => {
    const audio = audioRef.current
    if (!audio) return

    audio.muted = !isMuted
    setIsMuted(!isMuted)
  }

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current
    if (!audio) return

    const newVolume = Number(e.target.value)
    audio.volume = newVolume
    setVolume(newVolume)
    setIsMuted(newVolume === 0)
  }

  const skip = (seconds: number) => {
    const audio = audioRef.current
    if (!audio) return

    audio.currentTime = Math.max(0, Math.min(duration, audio.currentTime + seconds))
  }

  return (
    <div className="bg-studio-panel rounded-lg border border-studio-border p-4">
      <audio ref={audioRef} src={src || undefined} />

      <div className="mb-3">
        <canvas
          ref={canvasRef}
          width={600}
          height={80}
          onClick={handleCanvasClick}
          className="w-full h-20 rounded cursor-pointer"
          title="Click to seek, Shift+Click to select time for extend/crop"
        />
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => skip(-10)}
            disabled={!src}
            className="p-2 text-studio-muted hover:text-studio-text disabled:opacity-50 transition-colors"
            title="Skip back 10s"
          >
            <SkipBack className="w-5 h-5" />
          </button>

          <button
            onClick={togglePlay}
            disabled={!src}
            className="p-3 bg-studio-accent hover:bg-blue-600 text-white rounded-full disabled:opacity-50 transition-colors"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
          </button>

          <button
            onClick={() => skip(10)}
            disabled={!src}
            className="p-2 text-studio-muted hover:text-studio-text disabled:opacity-50 transition-colors"
            title="Skip forward 10s"
          >
            <SkipForward className="w-5 h-5" />
          </button>
        </div>

        <div className="text-sm text-studio-muted">
          {formatDuration(currentTime * 1000)} / {formatDuration(duration * 1000)}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={toggleMute}
            className="p-2 text-studio-muted hover:text-studio-text transition-colors"
          >
            {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={isMuted ? 0 : volume}
            onChange={handleVolumeChange}
            className="w-20 accent-studio-accent"
          />
        </div>
      </div>

      {onTimeSelect && (
        <p className="text-xs text-studio-muted mt-2 text-center">
          Shift+Click on waveform to select time for extend/crop
        </p>
      )}
    </div>
  )
}
