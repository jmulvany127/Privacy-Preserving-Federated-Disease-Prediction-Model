#!/bin/bash

# === Experiment Settings ===
EXPERIMENT_NAME="exp_ckks_params"
NUM_CLIENTS=2
SLEEP_BETWEEN_RUNS=10  # seconds
PORT=65432  # server port to check

# === Create logs directory ===
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# === Helper: Wait for port to become free ===
wait_for_port() {
  echo "Checking if port $PORT is available..."
  while lsof -i :$PORT -sTCP:LISTEN -t >/dev/null; do
    echo "Port $PORT is still in use. Waiting..."
    sleep 2
  done
}

# === Baseline values ===
BASELINE_DEGREE=8192
BASELINE_BITS="60,40,40,60"
BASELINE_SCALE=40

# === Sweep 1: poly_modulus_degree ===
POLY_DEGREES=(4096 8192 16384 32768)
SAFE_BITS=("30,20,30" "60,40,40,60" "60,40,40,60" "60,40,40,60")

for i in "${!POLY_DEGREES[@]}"; do
  DEGREE="${POLY_DEGREES[$i]}"
  BITS="${SAFE_BITS[$i]}"
  RUN_NAME="${EXPERIMENT_NAME}_polydeg${DEGREE}"

  echo ""
  echo "=== Running poly_modulus_degree = $DEGREE with coeff_mod_bit_sizes = $BITS ==="

  pkill -f server_LDP.py
  wait_for_port

  nohup python3 -u server_LDP.py \
    --num_clients $NUM_CLIENTS \
    --experiment_name "$RUN_NAME" \
    --use_he True \
    --poly_modulus_degree $DEGREE \
    --coeff_mod_bit_sizes "$BITS" \
    --global_scale_exp $BASELINE_SCALE \
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
  echo "Finished poly_modulus_degree config: $DEGREE"
  sleep $SLEEP_BETWEEN_RUNS
done

# === Sweep 2: coeff_mod_bit_sizes ===
COEFF_BIT_OPTIONS=("30,20,30" "40,30,30,40" "50,20,20,50" "40,20,20,40")
COEFF_DEGREES=(4096 8192 8192 8192)

for i in "${!COEFF_BIT_OPTIONS[@]}"; do
  BITS="${COEFF_BIT_OPTIONS[$i]}"
  DEGREE="${COEFF_DEGREES[$i]}"
  RUN_NAME="${EXPERIMENT_NAME}_coeffbits$i"

  echo ""
  echo "=== Running coeff_mod_bit_sizes = $BITS with poly_modulus_degree = $DEGREE ==="

  pkill -f server_LDP.py
  wait_for_port

  nohup python3 -u server_LDP.py \
    --num_clients $NUM_CLIENTS \
    --experiment_name "$RUN_NAME" \
    --use_he True \
    --poly_modulus_degree $DEGREE \
    --coeff_mod_bit_sizes "$BITS" \
    --global_scale_exp $BASELINE_SCALE \
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
  echo "Finished COEFF_BITS config: $BITS"
  sleep $SLEEP_BETWEEN_RUNS
done

# === Sweep 3: global_scale_exp ===
SCALE_EXPS=(40)
for SCALE in "${SCALE_EXPS[@]}"; do
  RUN_NAME="${EXPERIMENT_NAME}_scale${SCALE}"
  echo ""
  echo "=== Running global_scale_exp = $SCALE ==="

  pkill -f server_LDP.py
  wait_for_port

  nohup python3 -u server_LDP.py \
    --num_clients $NUM_CLIENTS \
    --experiment_name "$RUN_NAME" \
    --use_he True \
    --poly_modulus_degree $BASELINE_DEGREE \
    --coeff_mod_bit_sizes "$BASELINE_BITS" \
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
  echo "Finished SCALE config: $SCALE"
  sleep $SLEEP_BETWEEN_RUNS
done

echo ""
echo "=== All CKKS parameter sweep experiments completed ==="
