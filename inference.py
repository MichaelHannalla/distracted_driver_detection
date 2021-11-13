import cv2
import argparse
import numpy as np
import tensorflow as tf 
import time
from tensorflow.keras.models import model_from_json

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default= "models/distracted_driver_detector_v7") 
    parser.add_argument("--folder", default=None)
    parser.add_argument("--camera", default=None, type= int)
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    input_shape = (320, 240)

    class_names = ["c0 Safe driving", "c1 Texting (right hand)", "c2 Talking on the phone (right hand)", "c3 Texting (left hand)",
        "c4 Talking on the phone (left hand)", "c5 Operating the radio", "c6 Drinking", "c7 Reaching behind", "c8 Hair and makeup", 
        "c9 Talking to passenger(s)"]       # List of class names

    # load json and create model
    json_file = open(args.model_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(args.model_path+ ".h5")
    print("Loaded model from disk")

    if args.camera == None and args.folder == None and args.image == None:
        raise RuntimeError("Please specify the input to either being from a folder, camera, or image.")
    
    if not args.camera == None:
        cap = cv2.VideoCapture(args.camera)
        while cap.isOpened():
            tic = time.time()
            ret, frame = cap.read()
            if ret == True:
                frame_resized = cv2.resize(frame, input_shape)
                frame_resized_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                input = frame_resized_rgb[np.newaxis, :]
                predictions = model.predict(input)
                pred_label = np.argmax(predictions)

                print("Predicted Driver Status: {}".format(class_names[pred_label]))
                cv2.imshow("input frame", frame_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            toc = time.time()
            print("Inference Time: {}s".format(toc-tic))
            print("FPS: {}s".format(1/(toc-tic)))
    
    elif not args.image == None:
        frame = cv2.imread(args.image)
        tic = time.time()
        frame_resized = cv2.resize(frame, input_shape)
        frame_resized_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input = frame_resized_rgb[np.newaxis, ...]
        input = input[..., np.newaxis]
        predictions = model.predict(input)
        print(predictions)
        pred_label = np.argmax(predictions)

        print("Predicted Driver Status: {}".format(class_names[pred_label]))
        cv2.imshow("input frame", frame_resized)
    
        toc = time.time()
        print("Inference Time: {}s".format(toc-tic))
        cv2.waitKey(0)
            

    elif not args.folder == None:
        raise NotImplementedError("Still under development!")
 