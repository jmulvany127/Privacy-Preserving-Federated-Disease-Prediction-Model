import os
import numpy as np
import tensorflow as tf
import socket
import pickle
import struct
import time
from sklearn.model_selection import train_test_split
from matplotlib import image as img
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from b_CNN import *
from b_utils import *
import csv
import os
from datetime import datetime
import psutil
import threading
import tenseal as ts
import argparse
from datetime import datetime
import os
import csv


#Parse command line args
parser = argparse.ArgumentParser(description="Federated Server")
parser.add_argument('--num_clients', type=int, default=2, help='Number of federated clients')
parser.add_argument('--use_he', type=str2bool, default=True, help='Toggle homomorphic encryption (default: True)')
parser.add_argument('--experiment_name', type=str, default=None, help='Optional name of the experiment (used to group logs)')
parser.add_argument('--poly_modulus_degree', type=int, default=8192, help='CKKS poly modulus degree (default: 8192)')
parser.add_argument('--coeff_mod_bit_sizes', type=str, default='60,40,40,60',
                    help='Comma-separated bit sizes for CKKS coeff modulus (default: "60,40,40,60")')
parser.add_argument('--global_scale_exp', type=int, default=40,
                    help='Exponent for CKKS global scale (e.g., 40 for 2^40)')

args = parser.parse_args()

# set CKKS params
#Convert comma-separated string to list of ints for coeff mod 
coeff_mod_bit_sizes = list(map(int, args.coeff_mod_bit_sizes.split(',')))
poly_modulus_degree = args.poly_modulus_degree
global_scale = 2 ** args.global_scale_exp

#set clients and whether or not to use HE
NUM_CLIENTS = args.num_clients
USE_HE = args.use_he


#Create a new evaluation log folder with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.experiment_name:
    base_log_dir = os.path.join("evaluation_logs", args.experiment_name)
    log_dir = os.path.join(base_log_dir, f"log_server_{timestamp}")
else:
    log_dir = os.path.join("evaluation_logs", f"log_server_{timestamp}")
    
os.makedirs(log_dir, exist_ok=True)

# CSV file for server logging
server_log_path = os.path.join(log_dir, "server_log.csv")
with open(server_log_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Round", "Accuracy", "F1 Score", "Round Duration (s)",
        "Confusion Matrix", "Precision", "Recall", "Support"
    ])
    
    
#Save server evaluation results to CSV
def log_server_round(round_num, acc, f1, duration, conf_matrix, report_dict):
    with open(server_log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            round_num,
            round(acc, 4),
            round(f1, 4),
            round(duration, 2),
            conf_matrix.flatten().tolist(),
            round(report_dict["weighted avg"]["precision"], 4),
            round(report_dict["weighted avg"]["recall"], 4),
            report_dict["weighted avg"]["support"]
        ])


# handle communication with each client during one round
def handle_client_communication(client_conn, global_weights, updates_list, updates_lock, comm_stats, index):
    sent_size = send_data(client_conn, global_weights)
    comm_stats['total_sent'] += sent_size

    client_encrypted_update, received_size = receive_data(client_conn)
    if client_encrypted_update is not None:
        with updates_lock:
            updates_list.append(client_encrypted_update)
        comm_stats['total_received'] += received_size

# modified Helper Functions for Socket Communication with Size Tracking 
def send_data(sock, data):
    try:
        data_pickle = pickle.dumps(data)
        data_length = len(data_pickle)
        header = struct.pack('!I', data_length)
        sock.sendall(header)
        sock.sendall(data_pickle)
        return 4 + data_length  # 4 bytes header + data size
    except Exception as e:
        raise e


def receive_data(sock):
    try:
        raw_msglen = recvall(sock, 4)
        if not raw_msglen:
            return None, 0
        msglen = struct.unpack('!I', raw_msglen)[0]
        
        data_pickle = recvall(sock, msglen)
        if not data_pickle:
            return None, 0
        return pickle.loads(data_pickle), 4 + msglen
    except Exception as e:
        print("[Server] Exception in receive_data:", e)
        return None, 0


def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

#Initialize TenSEAL Context on the Server 
if USE_HE:
    ts_context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    ts_context.global_scale = global_scale
    ts_context.generate_galois_keys()
    serialized_context = ts_context.serialize(save_secret_key=False)
else:
    ts_context = None
    serialized_context = None




# Server Setup 
HOST = '127.0.0.1'
PORT = 65432
client_data, (test_files, test_labels) = load_raw_covid_data_for_federated(num_clients=NUM_CLIENTS)

# Server test set (unseen by any client)
X_test, y_test = load_images_from_paths(list(zip(test_files, test_labels)))

# Initialize global model
global_model = CNN()
global_model.set_initial_params()

# Federated training settings
rounds = 25
F1_THRESHOLD = 0.9

round_times = []
comm_stats = {
    'total_sent': 0,
    'total_received': 0,
    'sent_per_round': [],
    'received_per_round': []
}
resource_metrics = []  # List of dicts for each round

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(NUM_CLIENTS)
    print("Waiting for clients to connect...")

    #Accept client connections
    clients = []
    for _ in range(NUM_CLIENTS):
        conn, addr = s.accept()
        clients.append(conn)
        print(f"Connected to {addr}")
        if USE_HE:
            send_data(conn, serialized_context)

    #train for a number of rounds or until F1 threshold reached
    for round_num in range(1, rounds + 1):
        start_time = time.time()
        print(f"\n--- Round {round_num}/{rounds} ---")
        updates = []
        updates_lock = threading.Lock()
        round_sent = 0
        round_received = 0

        # Start resource sampling for this round
        cpu_stop_event = threading.Event()
        cpu_usage_list = []
        cpu_thread = threading.Thread(target=sample_cpu_usage, args=(cpu_stop_event, cpu_usage_list))

        mem_stop_event = threading.Event()
        mem_usage_list = []
        mem_avail_list = []
        mem_thread = threading.Thread(target=sample_memory_usage, args=(mem_stop_event, mem_usage_list, mem_avail_list))

        cpu_thread.start()
        mem_thread.start()

        global_weights = global_model.get_weights()

        #set up thread for each client
        threads = []
        for idx, client_conn in enumerate(clients):
            t = threading.Thread(target=handle_client_communication, args=(client_conn, global_weights, updates, updates_lock, comm_stats, idx))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        comm_stats['sent_per_round'].append(comm_stats['total_sent'])
        comm_stats['received_per_round'].append(comm_stats['total_received'])
        



        #Aggregation logic
        if updates:
            aggregated_weights = []
            num_clients = len(updates)
            #If use HE is set then use homomorphic aggregation carried out 
            if USE_HE:
                for idx in range(len(updates[0])):  # For each layer
                    agg = ts.ckks_vector_from(ts_context, updates[0][idx])
                    for client_upd in updates[1:]:
                        agg += ts.ckks_vector_from(ts_context, client_upd[idx])
                    agg *= (1.0 / num_clients)
                    decrypted_flat = agg.decrypt()

                    original_shape = global_weights[idx].shape
                    decrypted_flat = np.array(decrypted_flat[:np.prod(original_shape)])
                    aggregated_weights.append(decrypted_flat.reshape(original_shape))
            else:
                # Simple average over all client updates if no HE)
                aggregated_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*updates)]

            global_model.set_weights(aggregated_weights)



        # Evaluate global model
        y_pred_probs = global_model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        acc = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, digits=4)
        f1 = f1_score(y_test, y_pred, average='weighted')  
        
        print(f"Round {round_num} - Accuracy: {acc*100:.2f}%")
        print(f"F1 Score: {f1:.4f}")  
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)



        # Stop resource sampling for this round
        cpu_stop_event.set()
        mem_stop_event.set()
        cpu_thread.join()
        mem_thread.join()

        # Compute and save resource usage statistics for the round
        if cpu_usage_list:
            avg_cpu = sum(cpu_usage_list) / len(cpu_usage_list)
            peak_cpu = max(cpu_usage_list)
            min_cpu = min(cpu_usage_list)
        else:
            avg_cpu = peak_cpu = min_cpu = 0.0

        if mem_usage_list:
            avg_mem = sum(mem_usage_list) / len(mem_usage_list)
            peak_mem = max(mem_usage_list)
            min_mem = min(mem_usage_list)
        else:
            avg_mem = peak_mem = min_mem = 0.0

        if mem_avail_list:
            avg_avail = sum(mem_avail_list) / len(mem_avail_list)
            peak_avail = max(mem_avail_list)
            min_avail = min(mem_avail_list)
        else:
            avg_avail = peak_avail = min_avail = 0.0

        round_resource = {
            'avg_cpu': avg_cpu,
            'peak_cpu': peak_cpu,
            'min_cpu': min_cpu,
            'avg_mem': avg_mem,
            'peak_mem': peak_mem,
            'min_mem': min_mem,
            'avg_avail': avg_avail,
            'peak_avail': peak_avail,
            'min_avail': min_avail
        }
        resource_metrics.append(round_resource)

        end_time = time.time()
        round_duration = end_time - start_time
        round_times.append(round_duration)
        
        # Parse classification report to dictionary
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        # Log to CSV
        log_server_round(round_num, acc, f1, round_duration, conf_matrix, report_dict)
        
        # Early Stopping Check Based on F1 Score 
        if f1 >= F1_THRESHOLD:
            print(f"\n*** F1 threshold reached: {f1:.4f} >= {F1_THRESHOLD}. Stopping training early. ***")
            break  
        
        print(f"Round {round_num} duration: {round_duration:.2f} seconds")

    # Final Reporting 
    total_time = sum(round_times)
    average_time_per_round = total_time / len(round_times) if round_times else 0
    average_time_per_round_per_client = average_time_per_round / NUM_CLIENTS

    print("\n=== Timing Report ===")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average time per round: {average_time_per_round:.2f} seconds")
    print(f"Average time per round per client: {average_time_per_round_per_client:.2f} seconds")
    print("Round durations:")
    for idx, duration in enumerate(round_times, 1):
        print(f"  Round {idx}: {duration:.2f} seconds")

    total_transferred = comm_stats['total_sent'] + comm_stats['total_received']
    avg_sent_per_client_total = comm_stats['total_sent'] / NUM_CLIENTS
    avg_received_per_client_total = comm_stats['total_received'] / NUM_CLIENTS

    print("\n=== Communication Overhead Report ===")
    print("Summary of Communication:")
    print(f"  Total Transferred: {total_transferred:,} bytes")
    print(f"  Total Sent: {comm_stats['total_sent']:,} bytes")
    print(f"  Total Received: {comm_stats['total_received']:,} bytes")
    print(f"  Avg Sent per Client (Total): {avg_sent_per_client_total:,.2f} bytes")
    print(f"  Avg Received per Client (Total): {avg_received_per_client_total:,.2f} bytes")
    print(f"  Avg Sent per Client per Round: {avg_sent_per_client_total/len(round_times):,.2f} bytes")
    print(f"  Avg Received per Client per Round: {avg_received_per_client_total/len(round_times):,.2f} bytes")

    if resource_metrics:
        n = len(resource_metrics)
        avg_cpu_overall = sum(r['avg_cpu'] for r in resource_metrics) / n
        overall_peak_cpu = max(r['peak_cpu'] for r in resource_metrics)
        overall_min_cpu = min(r['min_cpu'] for r in resource_metrics)
        avg_peak_cpu = sum(r['peak_cpu'] for r in resource_metrics) / n
        avg_min_cpu = sum(r['min_cpu'] for r in resource_metrics) / n

        avg_mem_overall = sum(r['avg_mem'] for r in resource_metrics) / n
        overall_peak_mem = max(r['peak_mem'] for r in resource_metrics)
        overall_min_mem = min(r['min_mem'] for r in resource_metrics)
        avg_peak_mem = sum(r['peak_mem'] for r in resource_metrics) / n
        avg_min_mem = sum(r['min_mem'] for r in resource_metrics) / n

        avg_avail_overall = sum(r['avg_avail'] for r in resource_metrics) / n
        overall_peak_avail = max(r['peak_avail'] for r in resource_metrics)
        overall_min_avail = min(r['min_avail'] for r in resource_metrics)
        avg_peak_avail = sum(r['peak_avail'] for r in resource_metrics) / n
        avg_min_avail = sum(r['min_avail'] for r in resource_metrics) / n

        print("\n=== Resource Usage Report ===")
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
    else:
        print("\nNo resource usage metrics collected.")

    # Close client connections
    for conn in clients:
        conn.close()
        
    # Save experiment metadata
    metadata_path = os.path.join(log_dir, "experiment_metadata.txt")
    with open(metadata_path, 'w') as meta_file:
        meta_file.write(f"Experiment Timestamp: {timestamp}\n")
        meta_file.write(f"Use Homomorphic Encryption: {USE_HE}\n")
        meta_file.write(f"Number of Clients: {NUM_CLIENTS}\n")
        meta_file.write("CKKS Parameters:\n")
        meta_file.write(f"  Poly Modulus Degree: {poly_modulus_degree}\n")
        meta_file.write(f"  Coeff Mod Bit Sizes: {coeff_mod_bit_sizes}\n")
        meta_file.write(f"  Global Scale: 2^{args.global_scale_exp}\n")


    # Timing report
    with open(os.path.join(log_dir, "timing_report.txt"), 'w') as f:
        f.write(f"Total training time: {sum(round_times):.2f} seconds\n")
        f.write(f"Avg time per round: {sum(round_times)/len(round_times):.2f} seconds\n")
        for i, rt in enumerate(round_times):
            f.write(f"Round {i+1}: {rt:.2f} seconds\n")
            
    # Communication report
    with open(os.path.join(log_dir, "communication_overhead.txt"), 'w') as f:
        f.write(f"Total bytes sent: {comm_stats['total_sent']}\n")
        f.write(f"Total bytes received: {comm_stats['total_received']}\n")
        f.write(f"Sent per round: {comm_stats['sent_per_round']}\n")
        f.write(f"Received per round: {comm_stats['received_per_round']}\n")

    # Resource usage report
    with open(os.path.join(log_dir, "resource_usage.txt"), 'w') as f:
        for i, res in enumerate(resource_metrics, 1):
            f.write(f"Round {i}:\n")
            for k, v in res.items():
                f.write(f"  {k}: {v:.2f}\n")

    print("\nTraining complete.")