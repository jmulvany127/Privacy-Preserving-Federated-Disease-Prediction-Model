#!/bin/bash

# === Experiment Settings ===
EXPERIMENT_NAME="exp_dp_epsilon_sweep"
NUM_CLIENTS=2
REPEATS=8
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

# DP Epsilon Values 
EPSILONS=(0.1 1 5 10)

#  CKKS Default Parameters 
POLY_DEGREE=8192
COEFF_BITS="60,40,40,60"
SCALE=40

# Run Experiments 
for EPS in "${EPSILONS[@]}"; do
  for run_id in $(seq 1 $REPEATS); do
    RUN_NAME="exp_dp_eps${EPS}_r${run_id}"
    echo ""
    echo "=== Running Config: Epsilon=$EPS | Run $run_id ==="

    pkill -f a_server_LDP.py
    wait_for_port

    nohup python3 -u a_server_LDP.py \
      --num_clients $NUM_CLIENTS \
      --experiment_name "$RUN_NAME" \
      --use_he True \
      --poly_modulus_degree $POLY_DEGREE \
      --coeff_mod_bit_sizes "$COEFF_BITS" \
      --global_scale_exp $SCALE \
      > "$LOG_DIR/server_${RUN_NAME}.log" 2>&1 &

    SERVER_PID=$!
    sleep 15

    for ((j=0; j<$NUM_CLIENTS; j++)); do
      echo "Starting client $j..."
      nohup python3 a_client_LDP.py \
        --client_id $j \
        --num_clients $NUM_CLIENTS \
        --experiment_name "$RUN_NAME" \
        --use_dp True \
        --dp_epsilon $EPS \
        --dp_noise_type gaussian \
        --use_he True \
        > "$LOG_DIR/client_${RUN_NAME}_$j.log" 2>&1 &
    done

    wait $SERVER_PID
    echo "Finished Config: Epsilon=$EPS | Run $run_id"
    sleep $SLEEP_BETWEEN_RUNS
  done
done

echo ""
echo "DP Epsilon  Experiment Completed "
