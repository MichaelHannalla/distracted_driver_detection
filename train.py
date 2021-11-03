import argparse
from dataset import load_data
from model import DistractedDriverDetector
from utils import plot_metrics

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default= "state-farm-distracted-driver-detection/imgs/train") 
    parser.add_argument("--model_save_path", default= "models/")
    parser.add_argument("--input_shape", default= 128, type= int)
    parser.add_argument("--batch_size", default=32, type= int)
    parser.add_argument("--epochs", default=20, type= int)
    args = parser.parse_args()

    # Load the dataset
    input_shape = (args.input_shape, args.input_shape, 3) 
    batch_size = args.batch_size
    class_names = ["c0 Safe driving", "c1 Texting (right hand)", "c2 Talking on the phone (right hand)", "c3 Texting (left hand)",
        "c4 Talking on the phone (left hand)", "c5 Operating the radio", "c6 Drinking", "c7 Reaching behind", "c8 Hair and makeup", 
        "c9 Talking to passenger(s)"]       # List of class names
    state_farm_trainset, state_farm_valset = load_data(args.dataset_path, input_shape= input_shape, batch_size= batch_size, class_names= None)

    # Create an instance of our model and print the summary
    distracted_driver_detector = DistractedDriverDetector()
    distracted_driver_detector.set_dataset(state_farm_trainset, state_farm_valset, input_shape= input_shape, batch_size= batch_size, num_classes= len(class_names))    
    distracted_driver_detector.create_model()
    distracted_driver_detector.summary() 
    
    # Train the model
    history = distracted_driver_detector.train(epochs= args.epochs)

    # Save the model
    distracted_driver_detector.save_model(args.model_save_path)     

    # Show the performance curves 
    plot_metrics(history)

