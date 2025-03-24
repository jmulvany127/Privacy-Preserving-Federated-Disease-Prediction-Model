#!/bin/bash

NUM_CLIENTS=4

# Start the server with the correct number of clients
echo "Starting server with $NUM_CLIENTS clients..."
nohup python3 -u server_LDP.py --num_clients $NUM_CLIENTS > server.log 2>&1 &

# Wait for server to be ready
echo "Waiting for server to be ready..."
sleep 15

# Start clients
for ((i=0; i<$NUM_CLIENTS; i++)); do
    echo "Starting client $i..."
    nohup python3 client_LDP.py --client_id $i --num_clients $NUM_CLIENTS > client$i.log 2>&1 &
done

echo "✅ All processes started using nohup."

echo ""
echo "👉 Use these to view logs:"
echo "   tail -f server.log"
for ((i=0; i<$NUM_CLIENTS; i++)); do
    echo "   tail -f client$i.log"
done
