#!/bin/bash

#Experiment Settings 
EXPERIMENT_NAME="exp_ckks_perf_light_vs_heavy"
NUM_CLIENTS=2
REPEATS=10
SLEEP_BETWEEN_RUNS=60 # Seconds - allows ports to free up (issues here)
PORT=65432

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

#Helper: Wait for port to become free (doenst seem to be working - use sleep for safety)
wait_for_port() {
  echo "Checking if port $PORT is available..."
  while lsof -i :$PORT -sTCP:LISTEN -t >/dev/null; do
    echo "Port $PORT is still in use. Waiting..."
    sleep 2
  done
}

#  Configurations 
POLY_DEGREES=(8192 8192 16384)
COEFF_BITS=("40,30,30,40" "60,40,40,60" "60,40,40,60")
SCALES=(30 40 40)

# Run Experiments for eac config
for i in "${!POLY_DEGREES[@]}"; do
  DEGREE="${POLY_DEGREES[$i]}"
  BITS="${COEFF_BITS[$i]}"
  SCALE="${SCALES[$i]}"

  #run ten runs of each config 
  for run_id in $(seq 1 $REPEATS); do
    RUN_NAME="${EXPERIMENT_NAME}_deg${DEGREE}_run${run_id}"

    echo ""
    echo "=== Running Config: Degree=$DEGREE | Bits=$BITS | Scale=$SCALE | Run $run_id ==="
    
    #kill previous server to clear port 
    pkill -f a_server_LDP.py
    wait_for_port

    #start server
    nohup python3 -u a_server_LDP.py \
      --num_clients $NUM_CLIENTS \
      --experiment_name "$RUN_NAME" \
      --use_he True \
      --poly_modulus_degree $DEGREE \
      --coeff_mod_bit_sizes "$BITS" \
      --global_scale_exp $SCALE \
      > "$LOG_DIR/server_${RUN_NAME}.log" 2>&1 &

    SERVER_PID=$!
    sleep 15

    #start clients
    for ((j=0; j<$NUM_CLIENTS; j++)); do
      echo "Starting client $j..."
      nohup python3 a_client_LDP.py \
        --client_id $j \
        --num_clients $NUM_CLIENTS \
        --experiment_name "$RUN_NAME" \
        --use_he True \
        > "$LOG_DIR/client_${RUN_NAME}_$j.log" 2>&1 &
    done

    wait $SERVER_PID
    echo "Finished Config: Degree=$DEGREE | Run $run_id"
    sleep $SLEEP_BETWEEN_RUNS
  done
done

echo ""
echo "CKKS experiments completed "
