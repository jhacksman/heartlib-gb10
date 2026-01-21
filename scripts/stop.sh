#!/bin/bash
#
# Stop script for HeartLib Music Studio
# Usage: ./scripts/stop.sh
#

echo "=== Stopping HeartLib Music Studio ==="
echo ""

# Kill screen sessions
echo "Stopping backend 1..."
screen -S heartlib-backend-1 -X quit 2>/dev/null || true

echo "Stopping backend 2..."
screen -S heartlib-backend-2 -X quit 2>/dev/null || true

echo "Stopping frontend..."
screen -S heartlib-frontend -X quit 2>/dev/null || true

# Kill any remaining processes on the ports
for port in 8000 8001 3000; do
    pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
    fi
done

echo ""
echo "All services stopped."
