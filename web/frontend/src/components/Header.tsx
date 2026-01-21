import { Music, LogOut, User } from 'lucide-react'

interface HeaderProps {
  user: { name: string; credits: number } | null
  onLogout: () => void
}

export function Header({ user, onLogout }: HeaderProps) {
  return (
    <header className="bg-studio-panel border-b border-studio-border px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <Music className="w-8 h-8 text-studio-accent" />
        <h1 className="text-xl font-semibold text-studio-text">AI Music Studio</h1>
      </div>
      
      {user && (
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-studio-muted">
            <span className="text-studio-accent font-medium">{user.credits}</span>
            <span>credits</span>
          </div>
          
          <div className="flex items-center gap-2 text-studio-text">
            <User className="w-4 h-4" />
            <span>{user.name}</span>
          </div>
          
          <button
            onClick={onLogout}
            className="flex items-center gap-1 text-studio-muted hover:text-studio-text transition-colors"
          >
            <LogOut className="w-4 h-4" />
            <span>Logout</span>
          </button>
        </div>
      )}
    </header>
  )
}
