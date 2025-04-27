import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import time

# callback to measure and print time per epoch
class TimeHistory(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        print(f"Epoch {epoch + 1} took {elapsed_time:.2f} seconds")

# CNN Model Definition
class CNN:
    def __init__(self, weight_decimals=8):
        self.model = None
        self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)

    def set_weight_decimals(self, weight_decimals):
        return weight_decimals if 2 <= weight_decimals <= 8 else 8

    def set_initial_params(self):
        model = tf.keras.Sequential([
            #conv block 1
            tf.keras.layers.Conv2D(32, 5, activation='relu', input_shape=(128,128,1), 
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            #conv block 2
            tf.keras.layers.Conv2D(16, 5, activation='relu', 
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            #conv block 3
            tf.keras.layers.Conv2D(32, 5, activation='relu', 
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(2),
            
            #fully connected layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(200, activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer=optimizer,
            metrics=['accuracy']
        )
        self.model = model
    
    #Computes the avg and stdev of weight updatesbetween two sets of weights
    def compute_weight_update_norm_stats(self, old_weights, new_weights):
        diffs = [abs(new - old) for new, old in zip(new_weights, old_weights)]
        avg_diffs = [np.mean(diff) for diff in diffs]
        stds = [np.std(diff) for diff in diffs]
        shapes = [diff.shape for diff in diffs]
        return avg_diffs, stds, shapes


    # Computes the avg and stdevof gradient norms for a given mini-batch   
    def compute_gradient_norm_stats(self, x_batch, y_batch):
        x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch, training=True)
            loss = self.model.compiled_loss(y_batch, y_pred, regularization_losses=self.model.losses)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        grad_norms = [tf.norm(g) for g in gradients if g is not None]
        grad_norms_tensor = tf.stack(grad_norms)
        avg_grad_norm = tf.reduce_mean(grad_norms_tensor)
        std_grad_norm = tf.math.reduce_std(grad_norms_tensor)
        return avg_grad_norm.numpy(), std_grad_norm.numpy()

    def get_weights(self):
        weights = self.model.get_weights()
        return [np.round(w * 10**self.WEIGHT_DECIMALS) for w in weights]

    def set_weights(self, parameters):
        scaled_weights = [w / 10**self.WEIGHT_DECIMALS for w in parameters]
        self.model.set_weights(scaled_weights)

    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, acc
    
    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def fit(self, X_train, y_train, X_val, y_val, epochs=20, workers=4):
        time_callback = TimeHistory()  #Init the time tracking callback
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[time_callback]
        )
