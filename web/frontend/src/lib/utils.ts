import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000)
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
}

export function parseDuration(duration: string): number {
  const parts = duration.split(':')
  if (parts.length === 2) {
    return (parseInt(parts[0]) * 60 + parseInt(parts[1])) * 1000
  }
  return 30000
}
