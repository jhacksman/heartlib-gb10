import { useState } from "react"
import { Sparkles, Music, Mic, Clock, Sliders } from "lucide-react"

interface GenerationFormProps {
  onGenerate: (params: {
    prompt: string
    tags: string
    lyrics: string
    duration_ms: number
    flow_steps: number
    temperature: number
    cfg_scale: number
  }) => Promise<string>
  disabled?: boolean
}

export function GenerationForm({ onGenerate, disabled }: GenerationFormProps) {
  const [prompt, setPrompt] = useState("")
  const [tags, setTags] = useState("")
  const [lyrics, setLyrics] = useState("")
  const [duration, setDuration] = useState(30)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [flowSteps, setFlowSteps] = useState(10)
  const [temperature, setTemperature] = useState(1.0)
  const [cfgScale, setCfgScale] = useState(1.5)
  const [isInstrumental, setIsInstrumental] = useState(false)
  
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [generationError, setGenerationError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!prompt.trim() || isSubmitting) return

    setGenerationError(null)
    setIsSubmitting(true)
    
    try {
      let finalTags = tags
      if (isInstrumental) {
        const tagList = tags.split(",").map(t => t.trim()).filter(Boolean)
        if (!tagList.includes("instrumental")) {
          tagList.push("instrumental")
        }
        finalTags = tagList.join(",")
      }
      
      await onGenerate({
        prompt,
        tags: finalTags,
        lyrics: isInstrumental ? "" : lyrics,
        duration_ms: duration * 1000,
        flow_steps: flowSteps,
        temperature,
        cfg_scale: cfgScale,
      })
      
      // Keep form values after submission for easy iteration on lyrics/tags
    } catch (error) {
      setGenerationError(error instanceof Error ? error.message : "Generation failed")
    } finally {
      setIsSubmitting(false)
    }
  }

  const presetTags = [
    "pop", "rock", "jazz", "classical", "electronic", "country",
    "polka", "folk", "hip-hop", "r&b", "metal", "ambient",
    "happy", "sad", "energetic", "calm", "romantic", "piano"
  ]

  const addTag = (tag: string) => {
    const currentTags = tags.split(",").map(t => t.trim()).filter(Boolean)
    if (!currentTags.includes(tag)) {
      setTags([...currentTags, tag].join(","))
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-studio-panel rounded-lg border border-studio-border p-4 space-y-4">
      <h2 className="text-lg font-semibold text-studio-text flex items-center gap-2">
        <Sparkles className="w-5 h-5 text-studio-accent" />
        Create New Song
      </h2>

      {generationError && (
        <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-3 text-red-400 text-sm">
          {generationError}
        </div>
      )}

      <div>
        <label className="block text-sm text-studio-muted mb-1">
          <Music className="w-4 h-4 inline mr-1" />
          Song Name
        </label>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="My Awesome Track"
          className="w-full bg-studio-bg border border-studio-border rounded-lg p-2 text-studio-text placeholder-studio-muted focus:outline-none focus:border-studio-accent"
          disabled={isSubmitting}
        />
      </div>

      <div>
        <label className="block text-sm text-studio-muted mb-1">
          Style Tags (controls the music style)
        </label>
        <input
          type="text"
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          placeholder="polka,happy,accordion"
          className="w-full bg-studio-bg border border-studio-border rounded-lg p-2 text-studio-text placeholder-studio-muted focus:outline-none focus:border-studio-accent"
          disabled={isSubmitting}
        />
        <p className="text-xs text-studio-muted mt-1">Use short keywords separated by commas (e.g., polka,happy,accordion)</p>
        <div className="flex flex-wrap gap-1 mt-2">
          {presetTags.map((tag) => (
            <button
              key={tag}
              type="button"
              onClick={() => addTag(tag)}
              disabled={isSubmitting}
              className="px-2 py-1 text-xs bg-studio-bg border border-studio-border rounded hover:border-studio-accent text-studio-muted hover:text-studio-text transition-colors disabled:opacity-50"
            >
              {tag}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <label className="text-sm text-studio-muted">Type:</label>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setIsInstrumental(false)}
            disabled={isSubmitting}
            className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
              !isInstrumental
                ? "bg-studio-accent text-white border-studio-accent"
                : "bg-studio-bg text-studio-muted border-studio-border hover:border-studio-accent"
            }`}
          >
            With Vocals
          </button>
          <button
            type="button"
            onClick={() => setIsInstrumental(true)}
            disabled={isSubmitting}
            className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
              isInstrumental
                ? "bg-studio-accent text-white border-studio-accent"
                : "bg-studio-bg text-studio-muted border-studio-border hover:border-studio-accent"
            }`}
          >
            Instrumental
          </button>
        </div>
      </div>

      {!isInstrumental && (
        <div>
          <label className="block text-sm text-studio-muted mb-1">
            <Mic className="w-4 h-4 inline mr-1" />
            Lyrics (optional)
          </label>
          <textarea
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
            placeholder={"[Verse]\nYour lyrics here...\n\n[Chorus]\nThe chorus goes here..."}
            rows={4}
            className="w-full bg-studio-bg border border-studio-border rounded-lg p-3 text-studio-text placeholder-studio-muted focus:outline-none focus:border-studio-accent resize-none font-mono text-sm"
            disabled={isSubmitting}
          />
        </div>
      )}

      <div>
        <label className="block text-sm text-studio-muted mb-1">
          <Clock className="w-4 h-4 inline mr-1" />
          Duration: {duration}s
        </label>
        <input
          type="range"
          min={10}
          max={180}
          step={5}
          value={duration}
          onChange={(e) => setDuration(Number(e.target.value))}
          className="w-full accent-studio-accent"
          disabled={isSubmitting}
        />
        <div className="flex justify-between text-xs text-studio-muted">
          <span>10s</span>
          <span>180s</span>
        </div>
      </div>

      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-1 text-sm text-studio-muted hover:text-studio-text transition-colors"
        >
          <Sliders className="w-4 h-4" />
          {showAdvanced ? "Hide" : "Show"} Advanced Settings
        </button>

        {showAdvanced && (
          <div className="mt-3 space-y-3 p-3 bg-studio-bg rounded-lg">
            <div>
              <label className="block text-sm text-studio-muted mb-1">
                Quality (Flow Steps): {flowSteps}
              </label>
              <input
                type="range"
                min={5}
                max={20}
                value={flowSteps}
                onChange={(e) => setFlowSteps(Number(e.target.value))}
                className="w-full accent-studio-accent"
                disabled={isSubmitting}
              />
              <div className="flex justify-between text-xs text-studio-muted">
                <span>Fast (5)</span>
                <span>Quality (20)</span>
              </div>
            </div>

            <div>
              <label className="block text-sm text-studio-muted mb-1">
                Creativity (Temperature): {temperature.toFixed(1)}
              </label>
              <input
                type="range"
                min={0.5}
                max={2.0}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full accent-studio-accent"
                disabled={isSubmitting}
              />
              <div className="flex justify-between text-xs text-studio-muted">
                <span>Conservative (0.5)</span>
                <span>Creative (2.0)</span>
              </div>
            </div>

            <div>
              <label className="block text-sm text-studio-muted mb-1">
                Style Adherence (CFG): {cfgScale.toFixed(2)}
              </label>
              <input
                type="range"
                min={1.0}
                max={2.0}
                step={0.05}
                value={cfgScale}
                onChange={(e) => setCfgScale(Number(e.target.value))}
                className="w-full accent-studio-accent"
                disabled={isSubmitting}
              />
              <div className="flex justify-between text-xs text-studio-muted">
                <span>Loose (1.0)</span>
                <span>Strict (2.0)</span>
              </div>
            </div>
          </div>
        )}
      </div>

      <button
        type="submit"
        disabled={disabled || !prompt.trim() || isSubmitting}
        className="w-full bg-studio-accent hover:bg-blue-600 text-white font-medium py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
      >
        <Sparkles className="w-5 h-5" />
        {isSubmitting ? "Submitting..." : "Generate Song (10 credits)"}
      </button>
    </form>
  )
}
