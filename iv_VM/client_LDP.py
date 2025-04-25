import os
import numpy as np
import tensorflow as tf
import socket
import pickle
import struct
import argparse
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from matplotlib import image as img
from CNN import *
from utils import *
import warnings
import csv
import psutil
import tracemalloc
import gc
import time
import threading  # For sampling threads
import tenseal as ts
import pandas as pd
from datetime import datetime




def load_layer_sensitivities(csv_path):
    df = pd.read_csv(csv_path)
    round_to_layer_sens = {}

    for round_num, group in df.groupby('round'):
        round_to_layer_sens[int(round_num)] = {
            int(row['layer_index']): (row['avg_diffs']) for _, row in group.iterrows()
        }
    return round_to_layer_sens
layer_sensitivity_dict = load_layer_sensitivities("layer_update_avgs_1.csv")

# Define a helper function or mapping for dynamic threshold:
def get_dynamic_threshold(round_num):

    thresholds = {
        1: 100000000+3*244000000,
        2: 61000000+3*136000000,
        3: 45000000+3*1140000000,
        4: 37000000+3*93000000,
        5: 30000000+3*75000000,
        6: 30000000+3*75000000,
        7: 30000000+3*75000000,
        8: 25000000+3*65000000,
        9: 22000000+3*60000000,
        # From round 10 onward, threshold remains constant:
    }
    if round_num in thresholds:
        return thresholds[round_num]
    else:
        return 22000000+3*60000000  # This is the threshold for round 10 onward
    


# Function to sample Memory usage and available memory during training
def sample_memory_usage(stop_event, mem_usage_list, mem_avail_list):
    """
    Polls the process's memory usage and the system's available memory every second,
    appending the values (in MB) to mem_usage_list and mem_avail_list respectively,
    until stop_event is set.
    """
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        # Memory usage of this process (in MB)
        mem_usage = process.memory_info().rss / (1024 * 1024)
        # Available system memory (in MB)
        mem_avail = psutil.virtual_memory().available / (1024 * 1024)
        mem_usage_list.append(mem_usage)
        mem_avail_list.append(mem_avail)
        time.sleep(1)
        
def encrypt_weights(weights, ts_context):

    encrypted_weights = []
    for weight in weights:
        flat_weight = weight.flatten().tolist()
        enc_weight = ts.ckks_vector(ts_context, flat_weight)
        encrypted_weights.append(enc_weight.serialize())
    return encrypted_weights



def clip_weight_update(old, new, round_num):
    threshold = get_dynamic_threshold(round_num)
    update = new - old
    norm = np.linalg.norm(update)
    if norm > threshold:
        update = update * (threshold / norm)
    return old + update


def log_weight_update_stats(round_number, norms, stds, shapes, csv_path="layer_update_avgs.csv"):
    if args.client_id == 0:
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["round", "layer_index", "avg_diffs", "std_dev", "shape"])
            for i, (norm, std, shape) in enumerate(zip(norms, stds, shapes)):
                writer.writerow([round_number, i, norm, std, shape])


   
def add_dp_noise(weights, epsilon, round_sensitivities, round_num, num_clients, noise_type='laplace'):
    """
    Adds DP noise to each weight tensor using round-specific, layer-specific sensitivities.
    """
    noisy_weights = []
    print(f"\n=== DP Noise Injection (Round {round_num}) ===")
    print(f"Epsilon: {epsilon}, Noise Type: {noise_type}")

    for i, weight in enumerate(weights):
        sensitivity = round_sensitivities.get(round_num, {}).get(i, 1e-5)  # default to a small value
        noise_scale = (sensitivity) / (epsilon * np.sqrt(num_clients))

        #print(f"\n-- Layer {i} --")
        #print(f"Sensitivity: {sensitivity:.4f}")
        #print(f"Noise Scale: {noise_scale:.4f}")

        if noise_type == 'laplace':
            noise = np.random.laplace(loc=0.0, scale=noise_scale, size=weight.shape)
        elif noise_type == 'gaussian':
            noise = np.random.normal(loc=0.0, scale=noise_scale, size=weight.shape)
        else:
            raise ValueError("Unsupported noise type. Choose 'laplace' or 'gaussian'.")

        noisy_weight = weight + noise
        noisy_weights.append(noisy_weight)

    #print("=== End of Noise Injection ===\n")
    return noisy_weights



# Client Class
class Client:
    def __init__(self, X_train, y_train, X_val, y_val,
                 dp_epsilon, num_clients=2, dp_noise_type='gaussian',
                 use_dp=True, use_he=True):
        self.use_he = use_he
        self.model = CNN()
        self.model.set_initial_params()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.round_metrics = []  # List to store metrics for each round
        self.tenseal_context = None  
        
        # Store DP parameters
        self.use_dp = use_dp
        self.dp_epsilon = dp_epsilon
        self.num_clients = num_clients
        self.dp_noise_type = dp_noise_type
        
        self.current_round = 0
        self.update_threshold = None


    def update(self, global_weights):
        self.model.set_weights(global_weights)
        
        # Prepare for CPU usage sampling
        cpu_stop_event = threading.Event()
        cpu_usage_list = []
        cpu_thread = threading.Thread(target=sample_cpu_usage, args=(cpu_stop_event, cpu_usage_list))
        
        # Prepare for Memory usage sampling
        mem_stop_event = threading.Event()
        mem_usage_list = []
        mem_avail_list = []
        mem_thread = threading.Thread(target=sample_memory_usage, args=(mem_stop_event, mem_usage_list, mem_avail_list))
        
        # Start sampling threads
        cpu_thread.start()
        mem_thread.start()
        
        # Begin local training (this call is blocking)
        self.model.fit(self.X_train, self.y_train, self.X_val, self.y_val, epochs=5)
  
        # Stop the sampling threads after training completes
        cpu_stop_event.set()
        mem_stop_event.set()
        cpu_thread.join()
        mem_thread.join()
        
        # Compute CPU usage statistics
        if cpu_usage_list:
            avg_cpu = sum(cpu_usage_list) / len(cpu_usage_list)
            peak_cpu = max(cpu_usage_list)
            min_cpu = min(cpu_usage_list)
        else:
            avg_cpu = peak_cpu = min_cpu = 0.0
        
        # Compute Memory usage statistics (in MB)
        if mem_usage_list:
            avg_mem = sum(mem_usage_list) / len(mem_usage_list)
            peak_mem = max(mem_usage_list)
            min_mem = min(mem_usage_list)
        else:
            avg_mem = peak_mem = min_mem = 0.0
        
        # Compute Available Memory statistics (in MB)
        if mem_avail_list:
            avg_avail = sum(mem_avail_list) / len(mem_avail_list)
            peak_avail = max(mem_avail_list)
            min_avail = min(mem_avail_list)
        else:
            avg_avail = peak_avail = min_avail = 0.0
 
        self.current_round += 1

        plain_weights = self.model.get_weights()
        # Compute weight update norms: difference between new weights and global weights
        avg_diffs, stds, shapes = self.model.compute_weight_update_norm_stats(global_weights, plain_weights)
        log_weight_update_stats(self.current_round, avg_diffs, stds, shapes)
  
        # Clip each weight update (difference between new weights and global weights), 
        # based off emprical caclutions of weight update norm per round
        clipped_weights = []
        for i, (old_weight, new_weight) in enumerate(zip(global_weights, plain_weights)):
            clipped_weight = clip_weight_update(old_weight, new_weight, self.current_round)
            clipped_weights.append(clipped_weight)

            
        # Include gradient stats in the round metrics dictionary
        round_stats = {
            'avg_cpu': avg_cpu,
            'peak_cpu': peak_cpu,
            'min_cpu': min_cpu,
            'avg_mem': avg_mem,
            'peak_mem': peak_mem,
            'min_mem': min_mem,
            'avg_avail': avg_avail,
            'peak_avail': peak_avail,
            'min_avail': min_avail,
        }
        self.round_metrics.append(round_stats)
        
        print("Round Complete.")
        print(f"  CPU Usage - Avg: {avg_cpu:.2f}%, Peak: {peak_cpu:.2f}%, Min: {min_cpu:.2f}%")
        print(f"  Memory Usage - Avg: {avg_mem:.2f} MB, Peak: {peak_mem:.2f} MB, Min: {min_mem:.2f} MB")
        print(f"  Available Memory - Avg: {avg_avail:.2f} MB, Peak: {peak_avail:.2f} MB, Min: {min_avail:.2f} MB")

        # Log to CSV
        with open(client_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_round,
                avg_cpu, peak_cpu, min_cpu,
                avg_mem, peak_mem, min_mem,
                avg_avail, peak_avail, min_avail
            ])


        if self.use_dp:
            noisy_weights = add_dp_noise(
                clipped_weights,
                epsilon=self.dp_epsilon,
                round_sensitivities=layer_sensitivity_dict,
                round_num=self.current_round,
                num_clients=self.num_clients,
                noise_type=self.dp_noise_type
            )
        else:
            print("\n[Client] DP is disabled. Sending clipped weights without noise.")
            noisy_weights = clipped_weights

        
        # Encrypt the noisy weights before sending them to the server
        if self.use_he:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                encrypted_weights = encrypt_weights(noisy_weights, self.tenseal_context)
            return encrypted_weights
        else:
            print("[Client] HE is disabled. Sending raw weights.")
            return noisy_weights

    
   
    
    def print_final_report(self):
        if not self.round_metrics:
            print("No round metrics to report.")
            return
        
        n = len(self.round_metrics)
        # CPU metrics
        avg_cpu_overall = sum(r['avg_cpu'] for r in self.round_metrics) / n
        overall_peak_cpu = max(r['peak_cpu'] for r in self.round_metrics)
        overall_min_cpu = min(r['min_cpu'] for r in self.round_metrics)
        avg_peak_cpu = sum(r['peak_cpu'] for r in self.round_metrics) / n
        avg_min_cpu = sum(r['min_cpu'] for r in self.round_metrics) / n
        
        # Memory usage metrics
        avg_mem_overall = sum(r['avg_mem'] for r in self.round_metrics) / n
        overall_peak_mem = max(r['peak_mem'] for r in self.round_metrics)
        overall_min_mem = min(r['min_mem'] for r in self.round_metrics)
        avg_peak_mem = sum(r['peak_mem'] for r in self.round_metrics) / n
        avg_min_mem = sum(r['min_mem'] for r in self.round_metrics) / n
        
        # Available memory metrics
        avg_avail_overall = sum(r['avg_avail'] for r in self.round_metrics) / n
        overall_peak_avail = max(r['peak_avail'] for r in self.round_metrics)
        overall_min_avail = min(r['min_avail'] for r in self.round_metrics)
        avg_peak_avail = sum(r['peak_avail'] for r in self.round_metrics) / n
        avg_min_avail = sum(r['min_avail'] for r in self.round_metrics) / n
        
        print("\n==== Final Client Report ====")
        print("CPU Usage:")
        print(f"  Average of averages: {avg_cpu_overall:.2f}%")
        print(f"  Overall Peak: {overall_peak_cpu:.2f}%")
        print(f"  Overall Min: {overall_min_cpu:.2f}%")
        print(f"  Average per round Peak: {avg_peak_cpu:.2f}%")
        print(f"  Average per round Min: {avg_min_cpu:.2f}%")
        
        print("Memory Usage:")
        print(f"  Average of averages: {avg_mem_overall:.2f} MB")
        print(f"  Overall Peak: {overall_peak_mem:.2f} MB")
        print(f"  Overall Min: {overall_min_mem:.2f} MB")
        print(f"  Average per round Peak: {avg_peak_mem:.2f} MB")
        print(f"  Average per round Min: {avg_min_mem:.2f} MB")
        
        print("Available Memory:")
        print(f"  Average of averages: {avg_avail_overall:.2f} MB")
        print(f"  Overall Peak: {overall_peak_avail:.2f} MB")
        print(f"  Overall Min: {overall_min_avail:.2f} MB")
        print(f"  Average per round Peak: {avg_peak_avail:.2f} MB")
        print(f"  Average per round Min: {avg_min_avail:.2f} MB")
        print("==== End of Report ====")



# Socket Helper Functions
def send_data(sock, data):
    data_pickle = pickle.dumps(data)
    sock.sendall(struct.pack('!I', len(data_pickle)))
    sock.sendall(data_pickle)

def receive_data(sock):
    try:
        #print("[Client] Receiving header...")
        raw_msglen = recvall(sock, 4)
        if not raw_msglen:
            #print("[Client] No header received. Connection may be closed.")
            return None
        msglen = struct.unpack('!I', raw_msglen)[0]
        #print("[Client] Header indicates payload length:", msglen)
        
        data_pickle = recvall(sock, msglen)
        if not data_pickle:
            #print("[Client] Payload not fully received.")
            return None
        return pickle.loads(data_pickle)
    except Exception as e:
        #print("[Client] Exception in receive_data:", e)
        return None


def recvall(sock, n):

    data = bytearray()
    while len(data) < n:
        try:
            #print("[Client] Currently received:", len(data), "of", n, "expected")
            packet = sock.recv(n - len(data))
            #print("[Client] Packet received with length:", len(packet))
            if not packet:
                print("[Client] Connection closed by server.")
                return None
            data.extend(packet)
        except MemoryError as e:
            print("[Client] MemoryError during sock.recv: ", e)
            return None
        except Exception as e:
            print("[Client] Exception during sock.recv: ", e)
            return None
    return data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Client Setup
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default=None,
                    help="Optional name of the experiment (used to group logs)")
parser.add_argument('--use_dp', type=str2bool, default=True,
                    help="Toggle differential privacy noise addition (default: True)")
parser.add_argument('--use_he', type=str2bool, default=True,
                    help="Toggle homomorphic encryption (default: True)")
parser.add_argument('--client_id', type=int, required=True, choices=[0, 1,2,3,4,5,6,7,8,9])
parser.add_argument('--dp_epsilon', type=float, default=3.0,
                    help="Privacy budget epsilon for local differential privacy")
parser.add_argument('--num_clients', type=int, default=2,
                    help="Total number of clients in the system")
parser.add_argument('--dp_noise_type', type=str, default='gaussian', choices=['laplace', 'gaussian'],
                    help="Type of noise to use for DP")
args = parser.parse_args()

# Create unique evaluation folder with subfolder for each client
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.experiment_name:
    base_log_dir = os.path.join("evaluation_logs", args.experiment_name)
    os.makedirs(base_log_dir, exist_ok=True)
    log_dir = os.path.join(base_log_dir, f"log_clients_{timestamp}", f"client_{args.client_id}")
else:
    log_dir = os.path.join("evaluation_logs", f"log_clients_{timestamp}", f"client_{args.client_id}")

os.makedirs(log_dir, exist_ok=True)

# Path to CSV log file
client_log_path = os.path.join(log_dir, "metrics_log.csv")
with open(client_log_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Round",
        "Avg CPU (%)", "Peak CPU (%)", "Min CPU (%)",
        "Avg Memory (MB)", "Peak Memory (MB)", "Min Memory (MB)",
        "Avg Avail Mem (MB)", "Peak Avail Mem (MB)", "Min Avail Mem (MB)"
    ])



client_data, _ = load_raw_covid_data_for_federated(num_clients=args.num_clients)
client_files = client_data[args.client_id]
X_all, y_all = load_images_from_paths(client_files)

# Count label distribution
covid_count = int(np.sum(y_all))
noncovid_count = len(y_all) - covid_count

print(f"\n Client {args.client_id} Data Summary")
print(f"  Total samples: {len(y_all)}")
print(f"  COVID samples: {covid_count}")
print(f"  Non-COVID samples: {noncovid_count}")

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.1, stratify=y_all, random_state=42
)

print(f"  Training samples: {len(y_train)}")
print(f"    ↳ COVID: {int(np.sum(y_train))}, Non-COVID: {len(y_train) - int(np.sum(y_train))}")
print(f"  Validation samples: {len(y_val)}")
print(f"    ↳ COVID: {int(np.sum(y_val))}, Non-COVID: {len(y_val) - int(np.sum(y_val))}")
client = Client(X_train, y_train, X_val, y_val,
                dp_epsilon=args.dp_epsilon,
                num_clients=args.num_clients,
                dp_noise_type=args.dp_noise_type,
                use_dp=args.use_dp,
                use_he=args.use_he)

# === Metadata Logging (Only for Client 0) ===
if args.client_id == 0:
    meta_path = os.path.join(log_dir, "client_metadata.txt")
    with open(meta_path, 'w') as f:
        f.write("=== Client Experiment Metadata ===\n")
        f.write(f"Client ID: {args.client_id}\n")
        f.write(f"Total Clients: {args.num_clients}\n")
        f.write(f"Homomorphic Encryption (HE): {args.use_he}\n")

        if args.use_he and client.tenseal_context is not None:
            context = client.tenseal_context
            f.write("CKKS Parameters (Received):\n")
            f.write(f"  - poly_modulus_degree: {context.poly_modulus_degree()}\n")
            f.write(f"  - coeff_mod_bit_sizes: {context.coeff_modulus_sizes()}\n")
            f.write(f"  - global_scale: {context.global_scale()}\n")

        f.write(f"Differential Privacy (DP): {args.use_dp}\n")
        if args.use_dp:
            f.write(f"  - epsilon: {args.dp_epsilon}\n")
            f.write(f"  - noise_type: {args.dp_noise_type}\n")



# Connect to server
HOST = '127.0.0.1'
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print(f"Client {args.client_id} connected to server.")
    
    # Receive serialized TenSEAL context from the server and load it.
    if args.use_he:
        serialized_context = receive_data(s)
        tenseal_context = ts.context_from(serialized_context)
        client.tenseal_context = tenseal_context

    
    while True:
        # Receive global weights (plain weights for local training)
        global_weights = receive_data(s)
        if global_weights is None:
            break
        
        # Train and send encrypted update
        updated_weights = client.update(global_weights)
        send_data(s, updated_weights)

# Print final report after training is complete
client.print_final_report()

print(f"Client {args.client_id} disconnected.")
