// Primary backend URL (for auth, listing songs, etc.)
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// All backend URLs for parallel generation (2 versions like Suno)
const BACKEND_URLS = (import.meta.env.VITE_BACKEND_URLS || `${API_URL}`).split(',').map((url: string) => url.trim())

let authToken: string | null = localStorage.getItem('authToken')

export function setAuthToken(token: string | null) {
  authToken = token
  if (token) {
    localStorage.setItem('authToken', token)
  } else {
    localStorage.removeItem('authToken')
  }
}

export function getAuthToken(): string | null {
  return authToken
}

async function fetchWithAuth(url: string, options: RequestInit = {}, baseUrl: string = API_URL, handleAuthError: boolean = true) {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> || {}),
  }
  
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }
  
  const response = await fetch(`${baseUrl}${url}`, {
    ...options,
    headers,
  })
  
  // Only handle 401 for primary backend requests
  // Secondary backends may not have the session (in-memory storage)
  if (response.status === 401 && handleAuthError && baseUrl === API_URL) {
    setAuthToken(null)
    window.location.reload()
  }
  
  return response
}

export interface User {
  id: string
  email: string
  name: string
  credits: number
}

export interface AuthResponse {
  token: string
  user: User
}

export interface Song {
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
  // For dual-generation: which version (A, B, etc.) and which backend
  version?: string
  backendUrl?: string
}

export interface JobStatus {
  job_id: string
  status: string
  progress: number
  message: string
  output_url: string | null
  duration_ms: number | null
  created_at: string | null
}

export const api = {
  // Auth
  async signup(email: string, password: string, name: string = ''): Promise<AuthResponse> {
    const response = await fetch(`${API_URL}/api/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, name }),
    })
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Signup failed')
    }
    const data = await response.json()
    setAuthToken(data.token)
    return data
  },

  async login(email: string, password: string): Promise<AuthResponse> {
    const response = await fetch(`${API_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Login failed')
    }
    const data = await response.json()
    setAuthToken(data.token)
    return data
  },

  async logout(): Promise<void> {
    await fetchWithAuth('/api/auth/logout', { method: 'POST' })
    setAuthToken(null)
  },

  async getMe(): Promise<User> {
    const response = await fetchWithAuth('/api/auth/me')
    if (!response.ok) throw new Error('Failed to get user')
    return response.json()
  },

  // Songs
  async generateSong(params: {
    prompt: string
    tags?: string
    lyrics?: string
    duration_ms?: number
    flow_steps?: number
    temperature?: number
    cfg_scale?: number
  }): Promise<{ job_ids: Array<{ job_id: string; version: string; backendUrl: string }>; status: string; message: string }> {
    const formData = new FormData()
    formData.append('prompt', params.prompt)
    formData.append('tags', params.tags || '')
    formData.append('lyrics', params.lyrics || '')
    formData.append('duration_ms', String(params.duration_ms || 30000))
    formData.append('flow_steps', String(params.flow_steps || 10))
    formData.append('temperature', String(params.temperature || 1.0))
    formData.append('cfg_scale', String(params.cfg_scale || 1.25))

    // Call all backends in parallel for multiple versions (like Suno)
    const versionLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    const generatePromises = BACKEND_URLS.map(async (backendUrl: string, index: number) => {
      try {
        // Don't handle auth errors for secondary backends (they may not have the session)
        const response = await fetchWithAuth('/api/songs/generate', {
          method: 'POST',
          body: formData,
        }, backendUrl, backendUrl === API_URL)
        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.detail || 'Generation failed')
        }
        const data = await response.json()
        return {
          job_id: data.job_id,
          version: versionLabels[index] || String(index + 1),
          backendUrl,
        }
      } catch (error) {
        console.error(`Generation failed on backend ${backendUrl}:`, error)
        return null
      }
    })

    const results = await Promise.all(generatePromises)
    const successfulJobs = results.filter((r): r is { job_id: string; version: string; backendUrl: string } => r !== null)

    if (successfulJobs.length === 0) {
      throw new Error('All generation attempts failed')
    }

    return {
      job_ids: successfulJobs,
      status: 'pending',
      message: `Started ${successfulJobs.length} version(s)`,
    }
  },

  async extendSong(songId: string, params: {
    extend_from_ms: number
    extend_duration_ms?: number
    prompt?: string
    direction?: 'before' | 'after'
    flow_steps?: number
    temperature?: number
    cfg_scale?: number
  }): Promise<{ job_id: string; status: string; message: string }> {
    const formData = new FormData()
    formData.append('extend_from_ms', String(params.extend_from_ms))
    formData.append('extend_duration_ms', String(params.extend_duration_ms || 30000))
    formData.append('prompt', params.prompt || '')
    formData.append('direction', params.direction || 'after')
    formData.append('flow_steps', String(params.flow_steps || 10))
    formData.append('temperature', String(params.temperature || 1.0))
    formData.append('cfg_scale', String(params.cfg_scale || 1.25))

    const response = await fetchWithAuth(`/api/songs/${songId}/extend`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Extension failed')
    }
    return response.json()
  },

  async cropSong(songId: string, startMs: number, endMs: number): Promise<{ job_id: string; status: string; message: string }> {
    const formData = new FormData()
    formData.append('start_ms', String(startMs))
    formData.append('end_ms', String(endMs))

    const response = await fetchWithAuth(`/api/songs/${songId}/crop`, {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Crop failed')
    }
    return response.json()
  },

  async listSongs(): Promise<Song[]> {
    // Fetch songs from all backends and aggregate
    const versionLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    const fetchPromises = BACKEND_URLS.map(async (backendUrl: string, index: number) => {
      try {
        // Don't handle auth errors for secondary backends (they may not have the session)
        const response = await fetchWithAuth('/api/songs', {}, backendUrl, backendUrl === API_URL)
        if (!response.ok) return []
        const songs: Song[] = await response.json()
        // Add version label and backend URL to each song
        return songs.map(song => ({
          ...song,
          version: versionLabels[index] || String(index + 1),
          backendUrl,
        }))
      } catch {
        return []
      }
    })

    const results = await Promise.all(fetchPromises)
    const allSongs = results.flat()
    
    // Sort by created_at descending (newest first)
    return allSongs.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
  },

  async getSong(songId: string, backendUrl?: string): Promise<JobStatus> {
    const response = await fetchWithAuth(`/api/songs/${songId}`, {}, backendUrl || API_URL)
    if (!response.ok) throw new Error('Failed to get song')
    return response.json()
  },

  async deleteSong(songId: string, backendUrl?: string): Promise<void> {
    const response = await fetchWithAuth(`/api/songs/${songId}`, { method: 'DELETE' }, backendUrl || API_URL)
    if (!response.ok) throw new Error('Failed to delete song')
  },

  getDownloadUrl(songId: string, backendUrl?: string): string {
    return `${backendUrl || API_URL}/api/songs/${songId}/download`
  },

  async getAudioBlob(songId: string, backendUrl?: string): Promise<Blob> {
    const response = await fetchWithAuth(`/api/songs/${songId}/download`, {}, backendUrl || API_URL)
    if (!response.ok) throw new Error('Failed to fetch audio')
    return response.blob()
  },
}
