#!/bin/bash

NUM_CLIENTS=2
EXPERIMENT_NAME="exp_1"  # <-- Set your experiment name here 

# Start the server with the correct number of clients and experiment name
echo "Starting server with $NUM_CLIENTS clients..."
#nohup python3 -u server_LDP.py --num_clients $NUM_CLIENTS --experiment_name "$EXPERIMENT_NAME" > server.log 2>&1 &
nohup python3 -u server_LDP.py \
  --num_clients $NUM_CLIENTS \
  --experiment_name "$EXPERIMENT_NAME" \
  --poly_modulus_degree 8192 \
  --coeff_mod_bit_sizes "60,40,40,60" \
  --global_scale_exp 40 \
  > server.log 2>&1 &

# Wait for server to be ready
echo "Waiting for server to be ready..."
sleep 15

# Start clients
for ((i=0; i<$NUM_CLIENTS; i++)); do
    echo "Starting client $i..."
    nohup python3 client_LDP.py --client_id $i --num_clients $NUM_CLIENTS --experiment_name "$EXPERIMENT_NAME" > client$i.log 2>&1 &
done

echo "All processes started using nohup."

echo ""
echo "Use these to view logs:"
echo "  tail -f server.log"
for ((i=0; i<$NUM_CLIENTS; i++)); do
    echo "  tail -f client$i.log"
done
