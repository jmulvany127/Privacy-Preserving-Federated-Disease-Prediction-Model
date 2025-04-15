#!/bin/bash

# === Experiment Settings ===
EXPERIMENT_NAME="exp_baseline"
NUM_CLIENTS=2
REPEATS=10
SLEEP_BETWEEN_RUNS=60  # Seconds
PORT=65432  # Server port to check before each run

# Create logs directory if it doesn't exist
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# === Helper: Wait for port to become free ===
wait_for_port() {
  echo "Checking if port $PORT is available..."
  while lsof -i :$PORT -sTCP:LISTEN -t >/dev/null; do
    echo "Port $PORT is still in use. Waiting..."
    sleep 5
  done
}

# === Core Configurations (client-specific args) ===
CONFIGS=(
  "--use_he False --use_dp False"                   # FL
  "--use_he True  --use_dp False"                   # FL + HE
  "--use_he True  --use_dp True  --dp_epsilon 3.0"  # FL + HE + DP
  "--use_he False --use_dp True  --dp_epsilon 3.0"  # FL + DP
)

# === Start Experiment ===
for config_index in "${!CONFIGS[@]}"; do
  CLIENT_ARGS="${CONFIGS[$config_index]}"

  # Extract --use_he value from client args using regex
  if [[ "$CLIENT_ARGS" =~ (--use_he[[:space:]]+(True|False)) ]]; then
    SERVER_HE="${BASH_REMATCH[1]}"
  else
    SERVER_HE="--use_he False"  # Default fallback
  fi

  for run_id in $(seq 1 $REPEATS); do
    echo ""
    echo "=== Running config $config_index, run $run_id ==="
    echo "     Client Args: $CLIENT_ARGS"
    echo "     Server HE: $SERVER_HE"

    # Define a unique sub-experiment name
    RUN_NAME="${EXPERIMENT_NAME}_cfg${config_index}_run${run_id}"

    # Kill any leftover server process
    echo "Killing any existing server processes..."
    pkill -f server_LDP.py
    PID_ON_PORT=$(lsof -i tcp:$PORT -sTCP:LISTEN -t)
    if [[ -n "$PID_ON_PORT" ]]; then
      kill -9 $PID_ON_PORT
    fi

    # Wait until the port is fully released
    wait_for_port

    # Start server
    echo "Starting server..."
    nohup python3 -u server_LDP.py \
      --num_clients $NUM_CLIENTS \
      --experiment_name "$RUN_NAME" \
      $SERVER_HE \
      > "$LOG_DIR/server_${RUN_NAME}.log" 2>&1 &

    SERVER_PID=$!

    echo "Waiting for server to initialize..."
    sleep 15

    # Start clients
    for ((i=0; i<$NUM_CLIENTS; i++)); do
      echo "Starting client $i..."
      nohup python3 client_LDP.py \
        --client_id $i \
        --num_clients $NUM_CLIENTS \
        --experiment_name "$RUN_NAME" \
        $CLIENT_ARGS \
        > "$LOG_DIR/client_${RUN_NAME}_$i.log" 2>&1 &
    done

    echo "All clients started. Waiting for training to finish..."

    # Wait for server to finish
    wait $SERVER_PID
    echo "Server finished for config $config_index, run $run_id."

    echo "Sleeping before next run..."
    sleep $SLEEP_BETWEEN_RUNS
  done
done

echo ""
echo "=== All experiment runs completed ==="

echo "Starting post-baseline performance experiment..."
nohup bash run_ckks_experiment.sh > performance_experiment_output.log 2>&1 &

