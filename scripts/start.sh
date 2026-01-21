#!/bin/bash
#
# Startup script for HeartLib Music Studio
# Usage: ./scripts/start.sh
#
# This script:
# 1. Kills any processes on ports 8000, 8001, 3000
# 2. Starts 2 backend instances (ports 8000 and 8001)
# 3. Starts the frontend on port 3000
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== HeartLib Music Studio Startup ==="
echo ""

# Function to kill process on a specific port
kill_port() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
        sleep 1
    else
        echo "Port $port is free"
    fi
}

# 1. Kill existing processes on ports
echo "[1/3] Clearing ports..."
kill_port 8000
kill_port 8001
kill_port 3000

# Also kill any existing screen sessions for our services
screen -S heartlib-backend-1 -X quit 2>/dev/null || true
screen -S heartlib-backend-2 -X quit 2>/dev/null || true
screen -S heartlib-frontend -X quit 2>/dev/null || true
sleep 1

# 2. Start backend instances
echo ""
echo "[2/3] Starting backend instances..."

cd "$PROJECT_ROOT/web/backend"

# Backend 1 on port 8000
echo "  Starting backend 1 on port 8000..."
screen -dmS heartlib-backend-1 bash -c "uvicorn app.main:app --host 0.0.0.0 --port 8000 2>&1 | tee -a /tmp/heartlib-backend-1.log"

# Backend 2 on port 8001
echo "  Starting backend 2 on port 8001..."
screen -dmS heartlib-backend-2 bash -c "uvicorn app.main:app --host 0.0.0.0 --port 8001 2>&1 | tee -a /tmp/heartlib-backend-2.log"

# 3. Start frontend
echo ""
echo "[3/3] Starting frontend..."

cd "$PROJECT_ROOT/web/frontend"

# Serve the built frontend on port 3000
echo "  Starting frontend on port 3000..."
screen -dmS heartlib-frontend bash -c "npx serve -s dist -l 3000 2>&1 | tee -a /tmp/heartlib-frontend.log"

# Wait a moment for services to start
sleep 3

echo ""
echo "=== Startup Complete ==="
echo ""
echo "Services running:"
echo "  Backend 1: http://0.0.0.0:8000 (screen -r heartlib-backend-1)"
echo "  Backend 2: http://0.0.0.0:8001 (screen -r heartlib-backend-2)"
echo "  Frontend:  http://0.0.0.0:3000 (screen -r heartlib-frontend)"
echo ""
echo "Logs:"
echo "  tail -f /tmp/heartlib-backend-1.log"
echo "  tail -f /tmp/heartlib-backend-2.log"
echo "  tail -f /tmp/heartlib-frontend.log"
echo ""
echo "To stop all services: ./scripts/stop.sh"
