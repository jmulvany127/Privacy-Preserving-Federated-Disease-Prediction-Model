#!/bin/bash

# Start the server
echo "Starting server..."
python3 server_LDP.py &
SERVER_PID=$!

# Wait for the server to initialize
echo "Waiting for server to be ready..."
sleep 15

# Start Client 0 in a new terminal
echo "Starting client 0..."
gnome-terminal -- bash -c "python3 client_LDP.py --client_id 0; exec bash"

# Start Client 1 in a new terminal
echo "Starting client 1..."
gnome-terminal -- bash -c "python3 client_LDP.py --client_id 1; exec bash"

# Optionally: wait for the server to finish (press Ctrl+C to kill manually)
wait $SERVER_PID
