#!/bin/bash

# Start the server in a tmux session
tmux new-session -d -s server 'python3 server_LDP.py'

# Wait for the server to initialize
echo "Server started. Waiting for 15 seconds..."
sleep 15

# Start client 0 in a new tmux session
tmux new-session -d -s client0 'python3 client_LDP.py --client_id 0'

# Start client 1 in a new tmux session
tmux new-session -d -s client1 'python3 client_LDP.py --client_id 1'

echo "✅ All processes started in tmux sessions:"
echo "   - server"
echo "   - client
