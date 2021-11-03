import cv2
import argparse
import numpy as np
import tensorflow as tf 

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default= "models/distracted_driver_detector_v1.ckpt") 
    parser.add_argument("--folder", default=None)
    parser.add_argument("--camera", default=None, type= int)
    args = parser.parse_args()

    class_names = ["c0 Safe driving", "c1 Texting (right hand)", "c2 Talking on the phone (right hand)", "c3 Texting (left hand)",
        "c4 Talking on the phone (left hand)", "c5 Operating the radio", "c6 Drinking", "c7 Reaching behind", "c8 Hair and makeup", 
        "c9 Talking to passenger(s)"]       # List of class names

    model = tf.keras.models.load_model(args.model_path)

    if args.camera == None and args.folder == None:
        raise RuntimeError("Please specify the input to either being from a folder or a camera")
    
    if not args.camera == None:
        cap = cv2.VideoCapture(args.camera)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_resized = cv2.resize(frame, (128, 128))
                input = frame_resized.reshape(1, 128, 128, 3)
                predictions = model.predict(input)
                pred_label = np.argmax(predictions)

                print("Predicted Driver Status: {}".format(class_names[pred_label]))
                cv2.imshow("input frame", frame_resized)
                cv2.waitKey()
    
    elif not args.folder == None:
        raise NotImplementedError("Still under development!")
 