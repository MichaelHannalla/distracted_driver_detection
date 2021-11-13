import argparse
import tensorflow as tf

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default= "models/distracted_driver_detector_v7.ckpt")
    args = parser.parse_args()

    # Load the ckpt model
    model = tf.keras.models.load_model(args.ckpt_path)
    # serialize model to JSON
    model_json = model.to_json()
    with open(args.ckpt_path + "/../distracted_driver_detector_v7.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(args.ckpt_path + "/../distracted_driver_detector_v7.h5")
    print("Saved model to as json")