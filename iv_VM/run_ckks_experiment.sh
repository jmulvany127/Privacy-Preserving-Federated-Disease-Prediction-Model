#!/bin/bash

# === Experiment Settings ===
EXPERIMENT_NAME="exp_ckks_light_vs_heavy"
NUM_CLIENTS=2
REPEATS=10
SLEEP_BETWEEN_RUNS=60
PORT=65432

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

wait_for_port() {
  echo "Checking if port $PORT is available..."
  while lsof -i :$PORT -sTCP:LISTEN -t >/dev/null; do
    echo "Port $PORT is still in use. Waiting..."
    sleep 2
  done
}

# === Configurations ===
POLY_DEGREES=(16384)
COEFF_BITS=("60,40,40,60")
SCALES=(40)

# === Run Experiments ===
for i in "${!POLY_DEGREES[@]}"; do
  DEGREE="${POLY_DEGREES[$i]}"
  BITS="${COEFF_BITS[$i]}"
  SCALE="${SCALES[$i]}"

  for run_id in $(seq 1 $REPEATS); do
    #RUN_NAME="${EXPERIMENT_NAME}_deg${DEGREE}_run${run_id}"
    RUN_NAME="exp_heavy_r${run_id}"
    echo ""
    echo "=== Running Config: Degree=$DEGREE | Bits=$BITS | Scale=$SCALE | Run $run_id ==="

    pkill -f server_LDP.py
    wait_for_port

    nohup python3 -u server_LDP.py \
      --num_clients $NUM_CLIENTS \
      --experiment_name "$RUN_NAME" \
      --use_he True \
      --poly_modulus_degree $DEGREE \
      --coeff_mod_bit_sizes "$BITS" \
      --global_scale_exp $SCALE \
      > "$LOG_DIR/server_${RUN_NAME}.log" 2>&1 &

    SERVER_PID=$!
    sleep 15

    for ((j=0; j<$NUM_CLIENTS; j++)); do
      echo "Starting client $j..."
      nohup python3 client_LDP.py \
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
echo "=== Lightweight vs Heavyweight CKKS experiments completed ==="
