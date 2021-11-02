import tensorflow as tf
import matplotlib.pyplot as plt

def plot_metrics(history):
    losses = history.history["loss"]
    accuracies = history.history["accuracy"]
    plt.figure(0)
    plt.plot(losses, 'b')
    plt.suptitle("Loss versus Epochs")

    plt.figure(1)
    plt.plot(accuracies, 'r')
    plt.suptitle("Accuracy versus Epochs")
    plt.show()
