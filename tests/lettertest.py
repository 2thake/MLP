import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from MLP import MLP

def average_cross_entropy_loss(y_true, y_pred):
    """
    Used to calculate the error in the predictions
    """
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def one_hot_encode(letter):
    """
    Used to convert letters to one-hot
    """
    index = ord(letter) - ord('A')  # Convert 'A'-'Z' to 0-25
    one_hot = np.zeros(26)
    one_hot[index] = 1
    return one_hot

def one_hot_decode(one_hot_vector):
    """
    used to convert one-hot to letters
    """
    index = np.argmax(one_hot_vector)
    return chr(index + ord('A'))

def test_letters_problem():
    # read the data from the file
    header_names = [
        'letter', 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ]
    data = pd.read_csv('letter-recognition.txt', names=header_names)

    # split the data into inputs and targets
    X = data.drop('letter', axis=1)
    y = data['letter']

    # encode the data
    y = np.array([one_hot_encode(letter) for letter in y])

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # print a startup message
    print("Training the MLP on the UCI dataset...")

    # create an MLP and train it using the training data
    mlp = MLP(num_inputs=16, num_hidden=100, num_outputs=26, task='classification')
    errors = mlp.train(X_train.values, y_train, epochs=1000, learning_rate=0.005, batch_size=10, decay_rate=0.8, learning_update_interval=40)

    # use the newly trained MLP to predict the outputs for the test inputs
    y_pred = []
    for inputs, target in zip(X_test.values, y_test):
        output = mlp.forward(inputs)
        y_pred.append(output)

    # calculate and print the percentage of correctly predicted outputs
    accuracy_vec = [1 if one_hot_decode(a) == one_hot_decode(b) else 0 for a, b in zip(y_test, y_pred)]
    accuracy = sum(accuracy_vec) / len(accuracy_vec)
    print("Accuracy:", accuracy)

    # print the final error
    print(average_cross_entropy_loss(y_test, y_pred))

    # decode the targets and predictions
    y_true = [one_hot_decode(y) for y in y_test]
    y_pred_decoded = [one_hot_decode(y) for y in y_pred]

    # plot the error as a function of epoch
    plt.plot(errors)
    plt.title('Error During Training for Letter Recognition Problem')
    plt.xlabel('Epoch')
    plt.ylabel('Error (Cross-Entropy Loss)')
    plt.show()

    # create a confusion matrix and an alphabet from the targets and predictions
    labels=sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred_decoded, labels=labels)

    # plot the confusion matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Letter Recognition Confusion Matrix')
    plt.show()



if __name__ == "__main__":
    test_letters_problem()