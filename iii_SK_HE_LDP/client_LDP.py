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




import psutil
import tracemalloc
import gc
import time
import threading  # For sampling threads

import tenseal as ts

# Function to sample CPU usage during training
def sample_cpu_usage(stop_event, cpu_usage_list):
    """
    Polls CPU usage every second and appends the value to cpu_usage_list
    until stop_event is set.
    """
    while not stop_event.is_set():
        # This call waits for 1 second and returns the CPU usage percentage.
        usage = psutil.cpu_percent(interval=1)
        cpu_usage_list.append(usage)

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
def add_dp_noise(weights, epsilon, sensitivity, num_clients, noise_type='laplace'):
    """
    Applies local differential privacy noise to each weight matrix.
    
    Parameters:
      weights (list of np.ndarray): Model weights.
      epsilon (float): Privacy budget.
      sensitivity (float): Sensitivity value for the weight updates.
      num_clients (int): Total number of clients. Used to adjust noise scale.
      noise_type (str): 'laplace' or 'gaussian'.
    
    Returns:
      list of np.ndarray: Noisy weights.
    
    The noise scale is adjusted by a factor of 1/sqrt(num_clients) so that if there are
    more clients (and hence more averaging), the noise added per client can be lower.
    """
    noisy_weights = []
    noise_scale = sensitivity / (epsilon * np.sqrt(num_clients))
    for weight in weights:
        if noise_type == 'laplace':
            noise = np.random.laplace(loc=0.0, scale=noise_scale, size=weight.shape)
        elif noise_type == 'gaussian':
            noise = np.random.normal(loc=0.0, scale=noise_scale, size=weight.shape)
        else:
            raise ValueError("Unsupported noise type. Choose 'laplace' or 'gaussian'.")
        noisy_weights.append(weight + noise)
    return noisy_weights


# Client Class
class Client:
    def __init__(self, X_train, y_train, X_val, y_val,
                 dp_epsilon=1.0, dp_sensitivity=1.0, num_clients=2, dp_noise_type='laplace', max_gradient=0.963):
        # Pass max_gradient to CNN on creation
        self.model = CNN(max_gradient=max_gradient)
        self.model.set_initial_params()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.round_metrics = []  # List to store metrics for each round
        self.tenseal_context = None  
        
        # Store DP parameters
        self.dp_epsilon = dp_epsilon
        self.dp_sensitivity = dp_sensitivity
        self.num_clients = num_clients
        self.dp_noise_type = dp_noise_type
    
    

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
            
        sample_x = self.X_train[:32]
        sample_y = self.y_train[:32]

        
        # Compute gradient norm statistics on a sample batch
        avg_grad_norm, std_grad_norm = self.model.compute_gradient_norm_stats(sample_x, sample_y)
        #print(f"Average gradient norm for this round: {avg_grad_norm:.4f}, Standard deviation: {std_grad_norm:.4f}")

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
            'avg_grad_norm': avg_grad_norm,
            'std_grad_norm': std_grad_norm
        }
        self.round_metrics.append(round_stats)
        
        print("Round Complete.")
        print(f"  CPU Usage - Avg: {avg_cpu:.2f}%, Peak: {peak_cpu:.2f}%, Min: {min_cpu:.2f}%")
        print(f"  Memory Usage - Avg: {avg_mem:.2f} MB, Peak: {peak_mem:.2f} MB, Min: {min_mem:.2f} MB")
        print(f"  Available Memory - Avg: {avg_avail:.2f} MB, Peak: {peak_avail:.2f} MB, Min: {min_avail:.2f} MB")

        
        plain_weights = self.model.get_weights()
            # *** NEW: Apply local differential privacy noise ***
        noisy_weights = add_dp_noise(
            plain_weights,
            epsilon=self.dp_epsilon,
            sensitivity=self.dp_sensitivity,
            num_clients=self.num_clients,
            noise_type=self.dp_noise_type
        )
        
        # Encrypt the noisy weights before sending them to the server
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            encrypted_weights = encrypt_weights(noisy_weights, self.tenseal_context)


        return encrypted_weights
    
   
    
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
        
         # Print gradient update norms per round
        print("\n==== Gradient Update Norms per Round ====")
        for i, metrics in enumerate(self.round_metrics, start=1):
            print(f"Round {i}: Average gradient norm: {metrics['avg_grad_norm']:.4f}, Standard deviation: {metrics['std_grad_norm']:.4f}")
        # Gather per-round average gradient norms
        grad_means = [r['avg_grad_norm'] for r in self.round_metrics]
        overall_avg_grad = sum(grad_means) / n
        # Compute the standard deviation of the average gradient norms across rounds
        overall_std_grad = np.std(grad_means)
        
        print("\n==== Final Client Report ====")
        # (Existing CPU, Memory, and Available Memory reports)
        print("Gradient Update Norms:")
        print(f"  Average across rounds: {overall_avg_grad:.4f}")
        print(f"  Standard Deviation across rounds: {overall_std_grad:.4f}")
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



# Client Setup
parser = argparse.ArgumentParser()
parser.add_argument('--client_id', type=int, required=True, choices=[0, 1], help="Client ID (0, 1)")
parser.add_argument('--dp_epsilon', type=float, default=1.0,
                    help="Privacy budget epsilon for local differential privacy")
parser.add_argument('--dp_sensitivity', type=float, default=1.0,
                    help="Sensitivity for DP noise calculation")
parser.add_argument('--num_clients', type=int, default=2,
                    help="Total number of clients in the system")
parser.add_argument('--dp_noise_type', type=str, default='laplace', choices=['laplace', 'gaussian'],
                    help="Type of noise to use for DP")
args = parser.parse_args()

# Load and split data
X, y = load_raw_covid_data(limit=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

# Split into client-specific data (2 equal parts)
X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train, random_state=42)
X1_val, X2_val, y1_val, y2_val = train_test_split(X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)

# Assign dataset to the correct client
if args.client_id == 0:
    X_client, y_client = X1, y1
    X_val_client, y_val_client = X1_val, y1_val
elif args.client_id == 1:
    X_client, y_client = X2, y2
    X_val_client, y_val_client = X2_val, y2_val

client = Client(X_client, y_client, X_val_client, y_val_client,
                dp_epsilon=args.dp_epsilon,
                dp_sensitivity=args.dp_sensitivity,
                num_clients=args.num_clients,
                dp_noise_type=args.dp_noise_type)


# Connect to server
HOST = '127.0.0.1'
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print(f"Client {args.client_id} connected to server.")
    
    # Receive serialized TenSEAL context from the server and load it.
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
