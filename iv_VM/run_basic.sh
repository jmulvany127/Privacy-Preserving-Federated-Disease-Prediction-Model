#!/bin/bash

NUM_CLIENTS=2
EXPERIMENT_NAME="base"  

# Create logs directory if it doesn't exist
LOG_DIR="logs"
mkdir -p "$LOG_DIR"


# Start server w/ client num - set CKKS params here manually (can also turn off HE and DP)
echo "Starting server with $NUM_CLIENTS clients..."
nohup python3 -u a_server_LDP.py \
  --num_clients $NUM_CLIENTS \
  --experiment_name "$EXPERIMENT_NAME" \
  --poly_modulus_degree 8192 \
  --coeff_mod_bit_sizes "60,40,40,60" \
  --global_scale_exp 40 \
  --use_he True\
  > "$LOG_DIR/server_${EXPERIMENT_NAME}.log" 2>&1 &

# Wait for server to be ready
echo "Waiting for server to be ready..."
sleep 15

# Start clients
for ((i=0; i<$NUM_CLIENTS; i++)); do
    echo "Starting client $i..."
    nohup python3 a_client_LDP.py \
    --client_id $i \
    --num_clients $NUM_CLIENTS \
    --experiment_name "$EXPERIMENT_NAME" \
    --use_dp True\
    --use_he True\
    --dp_epsilon 3.0\
    > "$LOG_DIR/client_${EXPERIMENT_NAME}_$i.log" 2>&1 &
done

echo "All processes started using nohup."


