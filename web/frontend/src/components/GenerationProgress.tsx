import { useEffect, useState } from "react"
import { Loader2, Clock, Sparkles, CheckCircle, XCircle } from "lucide-react"

interface ProgressData {
  status: string
  progress: number
  message: string
  queue_position: number
  output_url: string | null
}

interface GenerationProgressProps {
  jobId: string
  token: string
  onComplete: () => void
  onError: (error: string) => void
}

export function GenerationProgress({ jobId, token, onComplete, onError }: GenerationProgressProps) {
  const [data, setData] = useState<ProgressData | null>(null)
  const [startTime] = useState(Date.now())
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000))
    }, 1000)
    return () => clearInterval(timer)
  }, [startTime])

  useEffect(() => {
    const apiUrl = import.meta.env.VITE_API_URL || "http://10.9.12.253:8000"
    let cancelled = false
    
    const pollProgress = async () => {
      while (!cancelled) {
        try {
          const response = await fetch(`${apiUrl}/api/songs/${jobId}`, {
            headers: {
              "Authorization": `Bearer ${token}`
            }
          })
          if (response.ok) {
            const jobData = await response.json()
            setData({
              status: jobData.status,
              progress: jobData.progress,
              message: jobData.message,
              queue_position: 0,
              output_url: jobData.output_url
            })
            
            if (jobData.status === "completed") {
              onComplete()
              break
            }
            if (jobData.status === "failed") {
              onError(jobData.message)
              break
            }
          }
        } catch (e) {
          console.error("Error polling progress:", e)
        }
        await new Promise(r => setTimeout(r, 1000))
      }
    }
    
    pollProgress()
    
    return () => {
      cancelled = true
    }
  }, [jobId, token, onComplete, onError])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
  }

  const getStatusIcon = () => {
    if (!data) return <Loader2 className="w-5 h-5 animate-spin text-studio-accent" />
    switch (data.status) {
      case "completed":
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case "failed":
        return <XCircle className="w-5 h-5 text-red-500" />
      case "pending":
        return <Clock className="w-5 h-5 text-yellow-500" />
      default:
        return <Sparkles className="w-5 h-5 animate-pulse text-studio-accent" />
    }
  }

  const getStatusText = () => {
    if (!data) return "Initializing..."
    if (data.queue_position > 0) {
      return `Queued (position ${data.queue_position})`
    }
    return data.message || data.status
  }

  const progressPercent = data?.progress ? Math.round(data.progress * 100) : 0

  return (
    <div className="bg-studio-panel rounded-lg border border-studio-border p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {getStatusIcon()}
          <span className="text-studio-text font-medium">
            {data?.status === "completed" ? "Complete!" : "Generating..."}
          </span>
        </div>
        <div className="flex items-center gap-1 text-studio-muted text-sm">
          <Clock className="w-4 h-4" />
          {formatTime(elapsed)}
        </div>
      </div>
      
      {/* Progress bar */}
      <div className="w-full bg-studio-bg rounded-full h-3 overflow-hidden">
        <div 
          className="h-full bg-gradient-to-r from-studio-accent to-blue-400 transition-all duration-300 ease-out"
          style={{ width: `${progressPercent}%` }}
        />
      </div>
      
      <div className="flex items-center justify-between text-sm">
        <span className="text-studio-muted">{getStatusText()}</span>
        <span className="text-studio-text font-mono">{progressPercent}%</span>
      </div>
    </div>
  )
}
