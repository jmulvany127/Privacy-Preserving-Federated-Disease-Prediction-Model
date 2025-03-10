import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from progressbar import progressbar
from skimage.transform import resize
from matplotlib import image as img
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# CNN Model Definition (as provided)
class CNN:
    def __init__(self, weight_decimals=8):
        self.model = None
        self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)

    def set_weight_decimals(self, weight_decimals):
        return weight_decimals if 2 <= weight_decimals <= 8 else 8

    def set_initial_params(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 5, activation='relu', input_shape=(128,128,1), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(16, 5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(32, 5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        self.class_weight = {0: 1, 1: 5}  # Increase weight for COVID class
        
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        self.model = model

    def fit(self, X_train, y_train, X_val, y_val, epochs=20, workers=4):
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            workers=workers
        )

    def get_weights(self):
        weights = self.model.get_weights()
        return [np.round(w * 10**self.WEIGHT_DECIMALS) for w in weights]

    def set_weights(self, parameters):
        scaled_weights = [w / 10**self.WEIGHT_DECIMALS for w in parameters]
        self.model.set_weights(scaled_weights)

    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, acc

# Data Loading Functions
def load_raw_covid_data(limit=500):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    covid_path = os.path.join(script_dir, "data", "covid")
    non_covid_path = os.path.join(script_dir, "data", "noncovid")
    
    # Load and process images
    def process_images(path, limit):
        images = [f for f in os.listdir(path) if f.endswith('.png')][:limit]
        data = np.empty((len(images), 128, 128, 1), dtype=np.float32)
        for i, f in enumerate(images):
            #print(f)
            img_data = img.imread(os.path.join(path, f))
            data[i] = resize(img_data, (128, 128, 1), anti_aliasing=True)
        return data
    
    covid_images = process_images(covid_path, limit)
    non_covid_images = process_images(non_covid_path, limit)
    
    X = np.concatenate([covid_images, non_covid_images])
    y = np.array([1]*len(covid_images) + [0]*len(non_covid_images), dtype=np.float32)
    return X, y

# Federated Components
class Client:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.model = CNN()
        self.model.set_initial_params()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def update(self, global_weights):
        self.model.set_weights(global_weights)
        self.model.fit(self.X_train, self.y_train, self.X_val, self.y_val, epochs=5)
        return self.model.get_weights()

# Server Class
class Server:
    def __init__(self, test_data):
        self.global_model = CNN()
        self.global_model.set_initial_params()
        self.X_test, self.y_test = test_data
        self.round = 0

    def aggregate(self, client_updates):
        avg_weights = [np.mean(weights, axis=0) for weights in zip(*client_updates)]
        return avg_weights

    def evaluate(self):
        loss, acc = self.global_model.evaluate(self.X_test, self.y_test)
        print(f"Round {self.round} - Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")

        # Generate predictions
        y_pred = np.argmax(self.global_model.model.predict(self.X_test), axis=1)
        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Compute precision, recall, and F1-score
        report = classification_report(self.y_test, y_pred, target_names=['Non-COVID', 'COVID'])
        print("Classification Report:")
        print(report)

    def federated_train(self, clients, rounds=10):
        for _ in range(rounds):
            self.round += 1
            updates = []
            print("this whats happening")
            for i, client in enumerate(clients, start=1):
                print(f"Training Client {i}...")
                updates.append(client.update(self.global_model.get_weights()))
            new_weights = self.aggregate(updates)
            self.global_model.set_weights(new_weights)
            self.evaluate()

# Load data and print split details
X, y = load_raw_covid_data(limit=500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

def print_data_distribution(name, y):
    unique, counts = np.unique(y, return_counts=True)
    print(f"{name} Size: {len(y)}, Class Distribution: {dict(zip(unique, counts))}")

print_data_distribution("Training Set", y_train)
print_data_distribution("Validation Set", y_val)
print_data_distribution("Test Set", y_test)

# Split data for two clients
X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train, random_state=42)
X1_val, X2_val, y1_val, y2_val = train_test_split(X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)

# Create Clients
client1 = Client(X1, y1, X1_val, y1_val)
client2 = Client(X2, y2, X2_val, y2_val)

# Start FL Process
server = Server((X_test, y_test))
server.federated_train([client1, client2], rounds=12)