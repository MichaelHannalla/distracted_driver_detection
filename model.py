import numpy as np
import tensorflow as tf 
from keras import callbacks

class DistractedDriverDetector():
    def __init__(self) -> None:
        pass

    def summary(self) -> None:
        """
            Prints model summary (layers I/O)
            Arguments:
                None
            returns:
                None
        """
        self.model.summary()

    def set_dataset(self, training_dataset, validation_dataset, input_shape, batch_size, num_classes) -> None:
        """
            Sets datasets and train/val splits into variables accessible across the class.
            Arguments:
                None
            returns:
                None
        """
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.num_classes = num_classes


    def create_model(self) -> None:
        """
            Creates the keras neural network model.
            Arguments:
                None
            returns:
                None
        """
        self.model = tf.keras.Sequential()

        # Input block
        self.model.add(tf.keras.layers.InputLayer(input_shape = self.input_shape))
        
        # Conv Block 1
        self.model.add(tf.keras.layers.Conv2D(filters= 8, kernel_size=3, padding="same", activation= "relu"))
        self.model.add(tf.keras.layers.Conv2D(filters= 8, kernel_size=3, padding="same", activation= "relu"))
        #self.model.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides=1, padding="same"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.2))

        # Conv Block 2
        self.model.add(tf.keras.layers.Conv2D(filters= 16, kernel_size=3, padding="same", activation= "relu"))
        self.model.add(tf.keras.layers.Conv2D(filters= 16, kernel_size=3, padding="same", activation= "relu"))
        #self.model.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 1, padding="same"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.2))        

        # Conv Block 3
        self.model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size=3, padding="same", activation= "relu"))
        self.model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size=3, padding="same", activation= "relu"))
        self.model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size=3, padding="same", activation= "relu"))
        #self.model.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 1, padding="same"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.2))

        # Conv Block 4
        self.model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size=3, padding="same", activation= "relu"))
        self.model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size=3, padding="same", activation= "relu"))
        self.model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size=3, padding="same", activation= "relu"))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 2, padding="same"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.2))

        # Conv Block 5
        self.model.add(tf.keras.layers.Conv2D(filters= 128, kernel_size=3, padding="same", activation= "relu"))
        self.model.add(tf.keras.layers.Conv2D(filters= 128, kernel_size=3, padding="same", activation= "relu"))
        self.model.add(tf.keras.layers.Conv2D(filters= 128, kernel_size=3, padding="same", activation= "relu"))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 2, padding="same"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.2))

        # Fully Connected Layers
        self.model.add(tf.keras.layers.Flatten())
        
        self.model.add(tf.keras.layers.Dense(256, activation= "relu"))    
        self.model.add(tf.keras.layers.Dense(256, activation= "relu"))
        self.model.add(tf.keras.layers.Dense(256, activation= "relu"))
        self.model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs):
        """
            Trains the keras neural network model
            Arguments:
                None
            returns:
                Function call that returns History object that has metrics evolution throughout epochs
        """
        earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience= 5, restore_best_weights= True)

        return self.model.fit(self.training_dataset, epochs= epochs,
            validation_data = self.validation_dataset, 
            callbacks = [earlystopping])    

    def save_model(self, model_save_path):
        """
            Saves the .ckpt keras model
            Arguments:
                model_save_path (string): model saving relative path
            returns:
                None
        """
        self.model.save(model_save_path + "distracted_driver_detector_v7.ckpt", overwrite= True)

    def get_model(self) -> tf.keras.Sequential:
        return self.model
