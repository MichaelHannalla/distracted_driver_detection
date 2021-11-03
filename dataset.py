import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.python.ops.gen_batch_ops import batch


def load_data(directory, input_shape, batch_size, color_mode = "rgb", class_names= None):
    """
        Function for loading the dataset from the directory of the folder that contains images 
        in such a structure that holds each class inside a directory
        Arguments:
            directory (string): file path of the dataset
            input_shape (tuple of ints): width and height of the image
            batch_size (int): desired batch size
        returns:
            training_dataset (tf.keras.Dataset): a dataset which has the training points and their labels
            validation_dataset (tf.keras.Dataset): a dataset which has the validation points and their labels
            
    """
    # dataset = tf.keras.preprocessing.image_dataset_from_directory(
    #     directory,
    #     labels="inferred",
    #     label_mode= "int",
    #     class_names= class_names,
    #     color_mode= color_mode,
    #     batch_size= batch_size,
    #     image_size= (input_shape[0], input_shape[1]),
    #     shuffle= True,
    #     seed= None,
    #     validation_split= None,
    #     subset= None,
    #     interpolation= "bilinear",
    #     follow_links= False,
    #     crop_to_aspect_ratio= False,
    # )

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)    

    training_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                    directory=directory,
                                                    shuffle=True,
                                                    target_size=(input_shape[0], input_shape[1]), 
                                                    subset="training",
                                                    class_mode='sparse')

    validation_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                    directory=directory,
                                                    shuffle=True,
                                                    target_size=(input_shape[0], input_shape[1]), 
                                                    subset="validation",
                                                    class_mode='sparse')
    return training_dataset, validation_dataset
