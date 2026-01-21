#!/bin/bash
#
# Deploy script for HeartLib Music Studio
# Usage: ./scripts/deploy.sh
#
# This script:
# 1. Pulls the latest code from git
# 2. Rebuilds the frontend
# 3. Restarts the backend server
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== HeartLib Music Studio Deploy ==="
echo ""

# 1. Pull latest code
echo "[1/4] Pulling latest code..."
git pull origin main

# 2. Install any new backend dependencies
echo ""
echo "[2/4] Installing backend dependencies..."
cd "$PROJECT_ROOT/web/backend"
pip install -e . -q

# 3. Rebuild frontend
echo ""
echo "[3/4] Rebuilding frontend..."
cd "$PROJECT_ROOT/web/frontend"
npm install --silent
npm run build

# 4. Restart backend server
echo ""
echo "[4/4] Restarting backend server..."

# Kill existing uvicorn process if running
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 2

# Start backend in background using screen (so it persists after SSH disconnect)
cd "$PROJECT_ROOT/web/backend"
screen -dmS heartlib-backend bash -c "uvicorn app.main:app --host 0.0.0.0 --port 8000 2>&1 | tee -a /tmp/heartlib-backend.log"

echo ""
echo "=== Deploy Complete ==="
echo ""
echo "Backend running in screen session 'heartlib-backend'"
echo "  - View logs: screen -r heartlib-backend"
echo "  - Or: tail -f /tmp/heartlib-backend.log"
echo ""
echo "Frontend built to: $PROJECT_ROOT/web/frontend/dist"
echo "  - Serve with: npx serve -s dist -l 3000"
echo "  - Or configure your web server to serve the dist folder"
