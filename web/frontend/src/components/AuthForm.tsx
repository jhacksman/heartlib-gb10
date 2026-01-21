import { useState } from 'react'
import { Music, Mail, Lock, User } from 'lucide-react'

interface AuthFormProps {
  onLogin: (email: string, password: string) => Promise<void>
  onSignup: (email: string, password: string, name: string) => Promise<void>
}

export function AuthForm({ onLogin, onSignup }: AuthFormProps) {
  const [isLogin, setIsLogin] = useState(true)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      if (isLogin) {
        await onLogin(email, password)
      } else {
        await onSignup(email, password, name)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-studio-bg flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="flex items-center justify-center gap-3 mb-8">
          <Music className="w-12 h-12 text-studio-accent" />
          <h1 className="text-3xl font-bold text-studio-text">AI Music Studio</h1>
        </div>

        <div className="bg-studio-panel rounded-lg border border-studio-border p-6">
          <div className="flex mb-6">
            <button
              onClick={() => setIsLogin(true)}
              className={`flex-1 py-2 text-center rounded-l-lg transition-colors ${
                isLogin
                  ? 'bg-studio-accent text-white'
                  : 'bg-studio-bg text-studio-muted hover:text-studio-text'
              }`}
            >
              Login
            </button>
            <button
              onClick={() => setIsLogin(false)}
              className={`flex-1 py-2 text-center rounded-r-lg transition-colors ${
                !isLogin
                  ? 'bg-studio-accent text-white'
                  : 'bg-studio-bg text-studio-muted hover:text-studio-text'
              }`}
            >
              Sign Up
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {!isLogin && (
              <div>
                <label className="block text-sm text-studio-muted mb-1">Name</label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-studio-muted" />
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Your name"
                    className="w-full bg-studio-bg border border-studio-border rounded-lg py-2 pl-10 pr-4 text-studio-text placeholder-studio-muted focus:outline-none focus:border-studio-accent"
                  />
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm text-studio-muted mb-1">Email</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-studio-muted" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  required
                  className="w-full bg-studio-bg border border-studio-border rounded-lg py-2 pl-10 pr-4 text-studio-text placeholder-studio-muted focus:outline-none focus:border-studio-accent"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm text-studio-muted mb-1">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-studio-muted" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter password"
                  required
                  minLength={6}
                  className="w-full bg-studio-bg border border-studio-border rounded-lg py-2 pl-10 pr-4 text-studio-text placeholder-studio-muted focus:outline-none focus:border-studio-accent"
                />
              </div>
            </div>

            {error && (
              <div className="text-red-400 text-sm bg-red-400/10 rounded-lg p-3">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-studio-accent hover:bg-blue-600 text-white font-medium py-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Please wait...' : isLogin ? 'Login' : 'Create Account'}
            </button>
          </form>

          <p className="text-center text-studio-muted text-sm mt-4">
            {isLogin ? "Don't have an account? " : 'Already have an account? '}
            <button
              onClick={() => setIsLogin(!isLogin)}
              className="text-studio-accent hover:underline"
            >
              {isLogin ? 'Sign up' : 'Login'}
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}
