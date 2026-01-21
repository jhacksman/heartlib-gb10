const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

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

function isNetworkError(error: unknown): boolean {
  if (error instanceof TypeError) {
    const message = error.message.toLowerCase()
    return message.includes('load failed') || 
           message.includes('failed to fetch') || 
           message.includes('network') ||
           message.includes('cors')
  }
  return false
}

function getNetworkErrorMessage(): string {
  return `Cannot connect to server at ${API_URL}. Please make sure the backend is running.`
}

async function safeFetch(url: string, options: RequestInit = {}): Promise<Response> {
  try {
    return await fetch(url, options)
  } catch (error) {
    if (isNetworkError(error)) {
      throw new Error(getNetworkErrorMessage())
    }
    throw error
  }
}

async function fetchWithAuth(url: string, options: RequestInit = {}) {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> || {}),
  }
  
  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }
  
  const response = await safeFetch(`${API_URL}${url}`, {
    ...options,
    headers,
  })
  
  if (response.status === 401) {
    setAuthToken(null)
    window.location.href = '/login'
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
  output_url: string | null
  created_at: string
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
    const response = await safeFetch(`${API_URL}/api/auth/signup`, {
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
    const response = await safeFetch(`${API_URL}/api/auth/login`, {
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
  }): Promise<{ job_id: string; status: string; message: string }> {
    const formData = new FormData()
    formData.append('prompt', params.prompt)
    formData.append('tags', params.tags || '')
    formData.append('lyrics', params.lyrics || '')
    formData.append('duration_ms', String(params.duration_ms || 30000))
    formData.append('flow_steps', String(params.flow_steps || 10))
    formData.append('temperature', String(params.temperature || 1.0))
    formData.append('cfg_scale', String(params.cfg_scale || 1.25))

    const response = await fetchWithAuth('/api/songs/generate', {
      method: 'POST',
      body: formData,
    })
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Generation failed')
    }
    return response.json()
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
    const response = await fetchWithAuth('/api/songs')
    if (!response.ok) throw new Error('Failed to list songs')
    return response.json()
  },

  async getSong(songId: string): Promise<JobStatus> {
    const response = await fetchWithAuth(`/api/songs/${songId}`)
    if (!response.ok) throw new Error('Failed to get song')
    return response.json()
  },

  async deleteSong(songId: string): Promise<void> {
    const response = await fetchWithAuth(`/api/songs/${songId}`, { method: 'DELETE' })
    if (!response.ok) throw new Error('Failed to delete song')
  },

  getDownloadUrl(songId: string): string {
    return `${API_URL}/api/songs/${songId}/download`
  },
}
