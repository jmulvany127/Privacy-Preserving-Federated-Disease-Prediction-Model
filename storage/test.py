# Standard Libraries
import importlib
import math
import os
import sys
import time
import warnings
from typing import List, Tuple

# Third Party Imports
import flwr as fl
import tensorflow as tf
from memory_profiler import memory_usage
from rlwe_xmkckks import RLWE, Rq
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Local Imports
from load_covid import *

# Get absolute paths to let a user run the script from anywhere
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.basename(current_directory)
working_directory = os.getcwd()
# Add parent directory to Python's module search path
sys.path.append(os.path.join(current_directory, '..'))
# Compare paths
if current_directory == working_directory:
    from cnn import CNN
    import utils
else:
    # Add current directory to Python's module search path
    CNN = importlib.import_module(f"{parent_directory}.cnn").CNN
    import utils
    
X_train, X_test, y_train, y_test = load_raw_covid_data(limit=100)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1) #, random_state=1
memory_usage_start = memory_usage()[0]



# dynamic settings
WEIGHT_DECIMALS = 8
model = CNN(WEIGHT_DECIMALS)
utils.set_initial_params(model)
params, _ = utils.get_flat_weights(model)
#print(params[0:20])

# Define the custom callback to print confusion matrix and predictions
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, print_freq=20):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.print_freq = print_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_freq == 0:
            # Get predictions for validation data
            y_pred = self.model.predict(self.X_val)
            print (y_val)
            print((y_pred > 0.5).astype(int))
  

            # Optionally print some predictions
   

# Initialize and fit the model with the custom callback
confusion_matrix_callback = ConfusionMatrixCallback(X_val, y_val)
print(y_val)
print(y_train)
model.fit(X_train, y_train, X_val, y_val, callback = confusion_matrix_callback, epochs=1000)
   

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Final Evaluation - Loss: {loss}, Accuracy: {accuracy}")