#!/bin/bash

# Start the server
echo "Starting server..."
nohup python3 -u server_LDP.py > server.log 2>&1 &


# Wait for server to be ready
echo "Waiting for server to be ready..."
sleep 15

# Start client 0
echo "Starting client 0..."
nohup python3 client_LDP.py --client_id 0 > client0.log 2>&1 &

# Start client 1
echo "Starting client 1..."
nohup python3 client_LDP.py --client_id 1 > client1.log 2>&1 &

echo "✅ All processes started using nohup."

echo ""
echo "👉 Use these to view logs:"
echo "   tail -f server.log"
echo "   tail -f client0.log"
echo "   tail -f client1.log"

