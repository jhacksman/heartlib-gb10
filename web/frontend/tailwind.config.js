/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'studio-bg': '#1a1d23',
        'studio-panel': '#22262e',
        'studio-border': '#2d3139',
        'studio-accent': '#3b82f6',
        'studio-text': '#e5e7eb',
        'studio-muted': '#9ca3af',
      },
    },
  },
  plugins: [],
}
