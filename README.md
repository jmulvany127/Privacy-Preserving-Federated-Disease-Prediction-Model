# README.md

## Project Overview
This repository contains the codebase for my MAI  thesis at Trinity College Dublin, titled:  
**"Building a Privacy-Preserving Disease Prediction Model using Federated Learning with Homomorphic Encryption, and Differential Privacy."**

---

## Repository Structure

```
THESIS_BU/
├── data/                   # COVID and non-COVID lung scan images
├── exp_1_Results/           # Experiment 1 notebooks and outputs
├── exp_2_Results/           # Experiment 2 notebooks and outputs
├── exp_3_Results/           # Experiment 3 notebooks and outputs
├── iv_VM/                   # Main implementation code 
└── requirements.txt         # List of required Python packages
```

---

## Directory Descriptions

### `data/`
This folder contains the **COVID-19** and **non-COVID** lung scan  used for **training**, **validation**, and **testing**.

---

### `exp_1_Results/`, `exp_2_Results/`, `exp_3_Results/`
Each of these folders contains **two Jupyter notebooks** for their respective experiments.  
The notebooks handle:
- Data processing
- Model evaluation
- Displaying of results

Instructions on how to run these notebooks will be provided later in this README.

---

### `iv_VM/`
This folder holds the **main bulk of the code** for the assignment.
A full breakdown of the `iv_VM/` folder will follow later.

---

### `requirements.txt`
This file contains a list of **all Python libraries** required to run the project.


## `iv_VM/` Directory Breakdown

### `evaluation_logs/`
Contains all of the data generated from each experiment run.  
When an experiment is run, a folder is created with the experiment name and run number.  
Within this folder:
- A **server folder** and a **folder for each client** are created.

Contents:
- **Client folders** contain:
  - `client_metadata.txt` – Shows the parameters the client was using for differential privacy (DP) ( for experiment tracking).
  - `metrics_log.csv` – Records CPU, memory, and available memory usage per training round.
- **Server folder** contains:
  - `server_log.csv` – Records training performance metrics (accuracy, F1 score etc) per round.
  - `resource_usage.txt` – Server CPU and memory usage across rounds.
  - `communication_overhead.txt` – Details total bytes sent and received per round.
  - `experiment_metadata.txt` – Records number of clients and CKKS parameters for that run.
  - `timing_report.txt` – Records how long each round took to complete.

---

### `logs/`
During each experiment, a log file is created for each client and the server, containing their terminal output.  
This is useful for tracking experiment progress or debugging if anything goes wrong.

---

### `a_client_LDP.py`
Contains the **Federated Learning client code**:
- Local model training
- Application of Differential Privacy
- Optional Homomorphic Encryption (using TenSEAL)
- Communication with the server

---

### `a_server_LDP.py`
Contains the **Federated Learning server code**:
- Coordinates model aggregation across clients
- Handles encrypted or plain weight updates
- Evaluates model performance each round


---

### `b_CNN.py`
Defines the **Convolutional Neural Network (CNN) model** architecture used for COVID detection.

---

### `b_utils.py`
Contains **utility functions** used by the client and server, such as:
- CPU and memory sampling
- dat laoding functions

---

### `experiment_output.log`
Captures the **overall output** of running the bash scripts for experiments.  
Useful for viewing the full experiment output in one place.

---


### `layer_update_averages_1.csv`
Experimental weight update data used for DP noise addition

---

### `run_basic.sh`
Bash script to run a **single basic experiment** with:
- Federated Learning
- Homomorphic Encryption
- Differential Privacy
enabled by default.

Runs just one round by default.

---

### `run_exp_1.sh`
Bash script to run **Experiment 1**:
- Tests and compares different configurations:
  - FL
  - FL + HE
  - FL + DP
  - FL + HE + DP
- By default, each configuration is tested 10 times (can be customized).

---

### `run_exp_2.sh`
Bash script to run **Experiment 2**:
- Tests different **CKKS parameter configurations**.
- Specific parameter settings (light, medium, heavy encryption) are described in the report.
- Number of runs and configurations are customizable.

---

### `run_exp_3.sh`
Bash script to run **Experiment 3**:
- Tests different **Differential Privacy (DP) epsilon values**.
- Specific epsilon values used are described in the report.

## Running an Experiment

1. **Ensure all requirements are installed**
   
   Install all necessary Python packages by running:
   
   ```bash
   pip install -r requirements.txt
   ```

2. **Navigate to the `iv_VM/` directory**

   ```bash
   cd iv_VM
   ```

3. **Run the bash script for the desired experiment using `nohup`**

   Each experiment has its own bash script.  
   Use `nohup` to run it in the background and redirect the output.

4. **Example command**

   Generalized version:

   ```bash
   nohup bash run_<experiment>.sh > experiment_output.log 2>&1 &
   ```

   Specific example for Experiment 2:

   ```bash
   nohup bash run_exp_2.sh > experiment_output.log 2>&1 &
   ```

5. **Monitor experiment progress**

   - The overall experiment output is stored in `experiment_output.log`.
   - Each individual client and server also writes logs into the `logs/` folder.

6. **After the experiment completes**

   - Move all generated experiment folders from `evaluation_logs/` into the appropriate results folder (`exp_1_Results/`, `exp_2_Results/`, `exp_3_Results/`) based on the experiment type.

7. **Analyze the results**

   - Open the corresponding Jupyter notebooks (`Training_Results.ipynb`, `Resource_Usage_Results.ipynb`) located in the results folders.
   - Run the cells to extract, visualize, and interpret the training and resource usage data.

8. **To terminate an experiment manually**

   If needed, you can kill an ongoing experiment using:

   ```bash
   pkill -f run_<experiment>.sh
   ```

   Then, to ensure all client and server processes are also stopped, run:

   ```bash
   pkill -f a_server_LDP.py
   pkill -f a_client_LDP.py
   ```

   Note:  
   After killing processes, you may need to wait **around 30 seconds** for the server port to fully release before starting a new experiment.
