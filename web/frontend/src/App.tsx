import { useState, useEffect, useCallback } from 'react'
import { api, User, Song, getAuthToken, setAuthToken } from './lib/api'
import { Header } from './components/Header'
import { AuthForm } from './components/AuthForm'
import { GenerationForm } from './components/GenerationForm'
import { SongList } from './components/SongList'
import { AudioPlayer } from './components/AudioPlayer'
import { SongActions } from './components/SongActions'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [user, setUser] = useState<User | null>(null)
  const [songs, setSongs] = useState<Song[]>([])
  const [selectedSong, setSelectedSong] = useState<Song | null>(null)
  const [selectedTimeMs, setSelectedTimeMs] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)

  // Check for existing auth on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = getAuthToken()
      if (token) {
        try {
          const userData = await api.getMe()
          setUser(userData)
          loadSongs()
        } catch {
          setAuthToken(null)
        }
      }
      setLoading(false)
    }
    checkAuth()
  }, [])

  const loadSongs = useCallback(async () => {
    try {
      const songList = await api.listSongs()
      setSongs(songList)
    } catch (error) {
      console.error('Failed to load songs:', error)
    }
  }, [])

  // Poll for song status updates
  useEffect(() => {
    if (!user) return

    const pollInterval = setInterval(async () => {
      const pendingSongs = songs.filter(s => s.status === 'pending' || s.status === 'processing')
      if (pendingSongs.length > 0) {
        await loadSongs()
      }
    }, 2000)

    return () => clearInterval(pollInterval)
  }, [user, songs, loadSongs])

  const handleLogin = async (email: string, password: string) => {
    const response = await api.login(email, password)
    setUser(response.user)
    loadSongs()
  }

  const handleSignup = async (email: string, password: string, name: string) => {
    const response = await api.signup(email, password, name)
    setUser(response.user)
    loadSongs()
  }

  const handleLogout = async () => {
    await api.logout()
    setUser(null)
    setSongs([])
    setSelectedSong(null)
  }

  const handleGenerate = async (params: {
    prompt: string
    tags: string
    lyrics: string
    duration_ms: number
    flow_steps: number
    temperature: number
    cfg_scale: number
  }) => {
    try {
      await api.generateSong(params)
      await loadSongs()
      // Refresh user to update credits
      const userData = await api.getMe()
      setUser(userData)
    } catch (error) {
      console.error('Generation failed:', error)
      throw error
    }
  }

  const handleExtend = async (params: {
    extend_from_ms: number
    extend_duration_ms: number
    direction: 'before' | 'after'
    prompt?: string
  }) => {
    if (!selectedSong) return

    try {
      await api.extendSong(selectedSong.id, params)
      await loadSongs()
      const userData = await api.getMe()
      setUser(userData)
      setSelectedTimeMs(null)
    } catch (error) {
      console.error('Extension failed:', error)
      throw error
    }
  }

  const handleCrop = async (startMs: number, endMs: number) => {
    if (!selectedSong) return

    try {
      const result = await api.cropSong(selectedSong.id, startMs, endMs)
      await loadSongs()
      // Select the new cropped song
      const newSongs = await api.listSongs()
      const newSong = newSongs.find(s => s.id === result.job_id)
      if (newSong) setSelectedSong(newSong)
    } catch (error) {
      console.error('Crop failed:', error)
      throw error
    }
  }

  const handleDeleteSong = async (songId: string) => {
    if (!confirm('Are you sure you want to delete this song?')) return

    try {
      await api.deleteSong(songId)
      if (selectedSong?.id === songId) {
        setSelectedSong(null)
      }
      await loadSongs()
    } catch (error) {
      console.error('Delete failed:', error)
    }
  }

  const handleSelectSong = (song: Song) => {
    setSelectedSong(song)
    setSelectedTimeMs(null)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-studio-bg flex items-center justify-center">
        <div className="text-studio-muted">Loading...</div>
      </div>
    )
  }

  if (!user) {
    return <AuthForm onLogin={handleLogin} onSignup={handleSignup} />
  }

  return (
    <div className="min-h-screen bg-studio-bg flex flex-col">
      <Header user={user} onLogout={handleLogout} />

      <main className="flex-1 flex gap-4 p-4 overflow-hidden">
        {/* Left Panel - Generation Form */}
        <div className="w-96 flex-shrink-0 overflow-y-auto">
          <GenerationForm
            onGenerate={handleGenerate}
            disabled={user.credits < 10}
          />
        </div>

        {/* Center Panel - Player and Actions */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
          {selectedSong ? (
            <>
              <div className="bg-studio-panel rounded-lg border border-studio-border p-4">
                <h2 className="text-lg font-semibold text-studio-text mb-1">
                  {selectedSong.name}
                </h2>
                <p className="text-sm text-studio-muted mb-4">{selectedSong.prompt}</p>
              </div>

              <AudioPlayer
                src={selectedSong.output_url ? `${API_URL}${selectedSong.output_url}` : null}
                duration_ms={selectedSong.duration_ms}
                onTimeSelect={setSelectedTimeMs}
                selectedTime={selectedTimeMs}
              />

              {selectedSong.status === 'completed' && selectedSong.output_url && (
                <SongActions
                  songId={selectedSong.id}
                  songName={selectedSong.name}
                  durationMs={selectedSong.duration_ms}
                  downloadUrl={api.getDownloadUrl(selectedSong.id)}
                  selectedTimeMs={selectedTimeMs}
                  onExtend={handleExtend}
                  onCrop={handleCrop}
                  authToken={getAuthToken()}
                />
              )}
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center bg-studio-panel rounded-lg border border-studio-border">
              <p className="text-studio-muted">Select a song to play and edit</p>
            </div>
          )}
        </div>

        {/* Right Panel - Song List */}
        <div className="w-80 flex-shrink-0 overflow-y-auto">
          <SongList
            songs={songs}
            selectedSongId={selectedSong?.id || null}
            onSelectSong={handleSelectSong}
            onDeleteSong={handleDeleteSong}
          />
        </div>
      </main>
    </div>
  )
}

export default App
