import { useState, useRef, useEffect, useCallback } from 'react'
import { Play, Pause, SkipBack, SkipForward, Volume2, VolumeX, Loader2 } from 'lucide-react'
import { formatDuration } from '../lib/utils'
import { api } from '../lib/api'

interface AudioPlayerProps {
  songId: string | null
  duration_ms: number
  onTimeSelect?: (timeMs: number) => void
  selectedTime?: number | null
}

// Extract waveform peaks from audio buffer
function extractWaveformPeaks(audioBuffer: AudioBuffer, numPeaks: number): number[] {
  const channelData = audioBuffer.getChannelData(0) // Use first channel
  const samplesPerPeak = Math.floor(channelData.length / numPeaks)
  const peaks: number[] = []

  for (let i = 0; i < numPeaks; i++) {
    const start = i * samplesPerPeak
    const end = Math.min(start + samplesPerPeak, channelData.length)
    
    let max = 0
    for (let j = start; j < end; j++) {
      const abs = Math.abs(channelData[j])
      if (abs > max) max = abs
    }
    peaks.push(max)
  }

  // Normalize peaks to 0-1 range
  const maxPeak = Math.max(...peaks, 0.01)
  return peaks.map(p => p / maxPeak)
}

export function AudioPlayer({ songId, duration_ms, onTimeSelect, selectedTime }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(duration_ms / 1000)
  const [volume, setVolume] = useState(1)
  const [isMuted, setIsMuted] = useState(false)
  const [blobUrl, setBlobUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [waveformPeaks, setWaveformPeaks] = useState<number[]>([])

  // Fetch audio with authentication, create blob URL, and extract waveform
  useEffect(() => {
    if (!songId) {
      setBlobUrl(null)
      setWaveformPeaks([])
      return
    }

    let cancelled = false
    setIsLoading(true)
    setError(null)

    api.getAudioBlob(songId)
      .then(async blob => {
        if (cancelled) return
        
        const url = URL.createObjectURL(blob)
        setBlobUrl(url)
        
        // Decode audio to extract waveform
        try {
          if (!audioContextRef.current) {
            audioContextRef.current = new AudioContext()
          }
          const arrayBuffer = await blob.arrayBuffer()
          const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer)
          const peaks = extractWaveformPeaks(audioBuffer, 100)
          if (!cancelled) {
            setWaveformPeaks(peaks)
          }
        } catch (err) {
          console.error('Failed to decode audio for waveform:', err)
          // Fall back to empty waveform (will show placeholder)
        }
        
        setIsLoading(false)
      })
      .catch(err => {
        if (cancelled) return
        console.error('Failed to load audio:', err)
        setError('Failed to load audio')
        setIsLoading(false)
      })

    return () => {
      cancelled = true
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl)
      }
    }
  }, [songId])

  // Clean up blob URL on unmount
  useEffect(() => {
    return () => {
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl)
      }
    }
  }, [blobUrl])

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
  }, [currentTime, selectedTime, duration, waveformPeaks])

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.fillStyle = '#22262e'
    ctx.fillRect(0, 0, width, height)

    // Draw waveform bars
    const barCount = 100
    const barWidth = width / barCount - 1
    const barGap = 1

    for (let i = 0; i < barCount; i++) {
      const x = i * (barWidth + barGap)
      
      // Use real waveform peaks if available, otherwise use placeholder
      let barHeight: number
      if (waveformPeaks.length > 0) {
        // Real waveform from audio data
        const peakIndex = Math.min(i, waveformPeaks.length - 1)
        barHeight = waveformPeaks[peakIndex] * 0.8 * height + height * 0.05
      } else {
        // Placeholder: generate pseudo-random height based on position
        const seed = Math.sin(i * 0.5) * Math.cos(i * 0.3) + Math.sin(i * 0.1)
        barHeight = Math.abs(seed) * 0.6 * height + height * 0.1
      }

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
  }, [currentTime, duration, selectedTime, waveformPeaks])

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
    if (!audio || !blobUrl) return

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
      <audio ref={audioRef} src={blobUrl || undefined} />

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
            disabled={!blobUrl || isLoading}
            className="p-2 text-studio-muted hover:text-studio-text disabled:opacity-50 transition-colors"
            title="Skip back 10s"
          >
            <SkipBack className="w-5 h-5" />
          </button>

          <button
            onClick={togglePlay}
            disabled={!blobUrl || isLoading}
            className="p-3 bg-studio-accent hover:bg-blue-600 text-white rounded-full disabled:opacity-50 transition-colors"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : isPlaying ? (
              <Pause className="w-5 h-5" />
            ) : (
              <Play className="w-5 h-5 ml-0.5" />
            )}
          </button>

          <button
            onClick={() => skip(10)}
            disabled={!blobUrl || isLoading}
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

      {error && (
        <p className="text-xs text-red-500 mt-2 text-center">
          {error}
        </p>
      )}

      {onTimeSelect && !error && (
        <p className="text-xs text-studio-muted mt-2 text-center">
          Shift+Click on waveform to select time for extend/crop
        </p>
      )}
    </div>
  )
}
